from datetime import date
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from app.core import catalog
from app.rag.chain import apply_fallback, build_llm, TOPIC_EXTRACT_PROMPT, generate_recall_question
from app.rag.graph import build_rag_graph
from app.rag.study_session import flatten_conversation, new_session
from app.schemas.rag import TopicList
from app.visualizer.visualizer import visualize_mindmap_text
from app.writer.writer import write_weekly_note, write_tilnote

MINDMAP_MAX_DOCS = 10  # 한 프롬프트에 붓는 문서 수 상한 


def _saved(path, doc_type: str) -> dict:
    return {"type": doc_type, "file_name": path.name}


def _match_documents(query: str | None) -> list[str]:
    if isinstance(query, str) and query.strip():
        needle = query.lower()
        return [
            row["source_path"]
            for row in catalog.list_documents()
            if needle in row["source_path"].lower() or needle in row["title"].lower()
        ]

    # query 없으면 "오늘" — dailynote/til 중 오늘 created_at인 것만
    today = date.today().isoformat()
    return [
        row["source_path"]
        for doc_type in ("dailynote", "til")
        for row in catalog.list_documents(doc_type=doc_type)
        if row["created_at"].startswith(today)
    ]


def make_tools(llm=None, qa_graph=None):
    """llm/qa_graph 주입 — 테스트가 임베딩·벡터스토어 없이 툴 표면을 검사할 수 있게."""
    llm = llm if llm is not None else build_llm()
    qa_graph = qa_graph if qa_graph is not None else build_rag_graph()
    topic_extractor = apply_fallback(llm, lambda m: m.with_structured_output(TopicList))

    @tool(parse_docstring=True)
    def answer_question(question: str) -> str:
        """Search the study document corpus to answer a question.

        Use this when the user asks about a concept, requests an explanation, or asks
        something like "what is X" or "how does X work".
        Do NOT use this when the user wants to record or write down what they studied today —
        use `write_daily` for that.

        Args:
            question: users question
        """
        result = qa_graph.invoke({"question": question})
        sources = ", ".join(result.get("sources", [])) or "없음"
        return f"{result['answer']}\n\n(출처: {sources})"

    @tool
    def write_daily(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Start a retrieval-practice session for what the user studied today.

        Use this when the user wants to record, organize, or write down what they learned in
        a study or lecture session today — e.g. "오늘 공부한 거 정리해줘", "write today's notes",
        or any description of what was studied today.
        Do NOT use this to answer questions or explain a concept — use `answer_question` for that.
        Do NOT use this for retrospectives about a project or task — use `write_til` for that.

        Call this once to start. Do NOT summarize the material yourself — the session asks the
        user to explain each topic in their own words.
        """
        conversation = flatten_conversation(state["messages"])
        messages = TOPIC_EXTRACT_PROMPT.invoke({"conversation": conversation}).to_messages()
        topics = topic_extractor.invoke(messages)
        session = new_session(topics.topics)

        if not session["pending"]:
            return Command(update={"messages": [
                ToolMessage("오늘 정리할 학습 주제를 찾지 못했어요. 뭘 공부했는지 조금 더 말해줄래요?",
                            tool_call_id=tool_call_id),
            ]})

        question = generate_recall_question(llm, session["pending"][0], conversation)
        return Command(update={
            "pending": session["pending"],
            "answered": session["answered"],
            "seedlings": session["seedlings"],
            "messages": [ToolMessage(question, tool_call_id=tool_call_id)],
        })

    @tool(parse_docstring=True)
    def write_weekly(
        tool_call_id: Annotated[str, InjectedToolCallId],
        as_of: str | None = None,
    ) -> Command:
        """Synthesize this week's daily notes (Monday through today) into one weekly summary and save it.

        Use this when the user wants a weekly review or summary of what they studied this week —
        e.g. "이번 주 정리해줘", "summarize this week's learning".
        Do NOT use this for a single day's notes — use `write_daily` for that.

        Args:
            as_of: ISO date (YYYY-MM-DD) to treat as "today" when picking the week.
                Only set this if the user explicitly names a different week; otherwise omit it
                and let it default to today.
        """
        path = write_weekly_note(as_of)
        if path is None:
            return Command(update={"messages": [
                ToolMessage("이번 주에 저장된 데일리노트가 없어서 주간노트를 만들지 못했어요.",
                            tool_call_id=tool_call_id),
            ]})
        return Command(update={
            "saved_documents": [_saved(path, "weeklynote")],
            "messages": [ToolMessage(f"주간노트 저장 완료: {path.name}", tool_call_id=tool_call_id)],
        })

    @tool(parse_docstring=True)
    def write_til(
        tool_call_id: Annotated[str, InjectedToolCallId],
        what: str,
        learned: str,
        troubleshooting: str,
        reflection: str,
        actionplan: str,
        keywords: list[str] | str | None = None,
    ) -> Command:
        """Write and save a TIL-style (Today I Learned) retrospective note.

        Use this when the user wants to record a retrospective, reflection, or "what I learned
        today" entry about a project, task, or work session — e.g. "오늘 회고 남겨줘", "write a TIL".
        Do NOT use this for lecture or course study notes — use `write_daily` for that.

        Extract every value only from what the user explicitly said. Never invent a problem,
        reflection, or action the user did not mention — leave the field empty instead of guessing.

        Args:
            what: A brief account of the situation or work, as described by the user.
            learned: The core takeaway from the session.
            troubleshooting: The problem encountered and how it was resolved. Empty string if none mentioned.
            reflection: The user's reflection, as stated. Do not expand it beyond light polishing.
            actionplan: The next action, as stated by the user. Empty string if none mentioned.
            keywords: Short keywords, only from terms the user actually used, as a list of strings.
        """
        if isinstance(keywords, str):
            keywords = [c.strip() for c in keywords.split(",") if c.strip()]
        path = write_tilnote(what, learned, troubleshooting, reflection, actionplan, keywords or [])
        return Command(update={
            "saved_documents": [_saved(path, "til")],
            "messages": [ToolMessage(f"TIL 저장 완료: {path.name}", tool_call_id=tool_call_id)],
        })

    @tool(parse_docstring=True)
    def visualize_mindmap(
        tool_call_id: Annotated[str, InjectedToolCallId],
        query: str | None = None,
    ) -> Command:
        """Generate a mindmap recap of the user's study notes and return its plaintext for display.

        Use this when the user wants a visual recap/mindmap of what they've studied — e.g.
        "오늘 배운 거 마인드맵으로 보여줘", "week-02 마인드맵으로 정리해줘", "Docker 관련 내용 마인드맵으로".
        Do NOT use this to answer a question or explain a concept — use `answer_question` for that.

        Args:
            query: What to visualize, in the user's own words — a filename fragment (e.g. "week-02")
                or a topic keyword. Leave this empty when the user just says "오늘"/"today" with no
                specific document named — it then defaults to today's daily notes.
        """
        documents = _match_documents(query)
        if not documents:
            return Command(update={"messages": [
                ToolMessage("해당하는 문서를 찾지 못했어요.", tool_call_id=tool_call_id),
            ]})

        truncated = len(documents) > MINDMAP_MAX_DOCS
        documents = documents[:MINDMAP_MAX_DOCS]
        plaintext = visualize_mindmap_text(documents)
        note = f" (조건에 맞는 문서가 많아 {MINDMAP_MAX_DOCS}건만 사용)" if truncated else ""

        return Command(update={
            "mindmaps": [plaintext],
            "messages": [ToolMessage(f"마인드맵 생성 완료 ({len(documents)}건){note}",
                                     tool_call_id=tool_call_id)],
        })

    return [answer_question, write_daily, write_weekly, write_til, visualize_mindmap]
