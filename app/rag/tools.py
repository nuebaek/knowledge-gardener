from langchain_core.tools import tool
from app.rag.graph import build_rag_graph
from app.writer.writer import write_daily_note, write_weekly_note, write_tilnote
from app.visualizer.visualizer import visualize_mindmap_img

from app.core import catalog
from datetime import date

def make_tools():
    qa_graph = build_rag_graph()

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

    @tool(parse_docstring=True)
    def write_daily(topic: str, learned: str, related_concepts: list[str] | str | None = None) -> str:
        """Write and save a structured daily learning note about what the user studied today.
        Use this when the user wants to record, organize, or write down what they learned in
        a study or lecture session today — e.g. "오늘 공부한 거 정리해줘", "write today's notes",
        or any description of what was studied today.
        Do NOT use this to answer questions or explain a concept — use `answer_question` for that.
        Do NOT use this for retrospectives about a project or task — use `write_til` for that.

        Extract every value only from what the user explicitly said in this conversation.
        Never invent, infer, or add a topic or concept the user did not actually mention.

        Args:
            topic: The main subject studied, as stated by the user.
            learned: The user's own explanations collected during the retrieval conversation,
                concatenated verbatim — including explicit notes on what the user could not yet
                explain. Do not summarize, rewrite, or complete them before passing.
            related_concepts: Specific terms or concepts the user explicitly named or explained,
                as a list of strings. Leave empty if none were mentioned — do not add concepts
                on your own.
        """
        if isinstance(related_concepts, str):
            related_concepts = [c.strip() for c in related_concepts.split(",") if c.strip()]
        return write_daily_note(topic, learned, related_concepts)

    @tool(parse_docstring=True)
    def write_weekly(as_of: str | None = None) -> str:
        """Synthesize this week's daily notes (Monday through today) into one weekly summary and save it.

        Use this when the user wants a weekly review or summary of what they studied this week —
        e.g. "이번 주 정리해줘", "summarize this week's learning".
        Do NOT use this for a single day's notes — use `write_daily` for that.

        Args:
            as_of: ISO date (YYYY-MM-DD) to treat as "today" when picking the week.
                Only set this if the user explicitly names a different week; otherwise omit it
                and let it default to today.
        """
        return write_weekly_note(as_of)
    
    @tool(parse_docstring=True)
    def write_til(
        what: str,
        learned: str,
        troubleshooting: str,
        reflection: str,
        actionplan: str,
        keywords: list[str] | str | None = None,
    ) -> str:
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
        return write_tilnote(what, learned, troubleshooting, reflection, actionplan, keywords or [])

    @tool(parse_docstring=True)
    def visualize_mindmap(query: str | None = None) -> str:
        """Generate a mindmap recap of the user's study notes and return its plaintext for display.

        Use this when the user wants a visual recap/mindmap of what they've studied — e.g.
        "오늘 배운 거 마인드맵으로 보여줘", "week-02 마인드맵으로 정리해줘", "Docker 관련 내용 마인드맵으로".
        Do NOT use this to answer a question or explain a concept — use `answer_question` for that.

        Args:
            query: What to visualize, in the user's own words — a filename fragment (e.g. "week-02")
                or a topic keyword. Leave this empty when the user just says "오늘"/"today" with no
                specific document named — it then defaults to today's daily notes.
        """
        documents = []
        if isinstance(query, str) and query.strip():
        # 파일명/제목에 query가 들어간 문서를 doc_type 무관하게 찾는다            
            rows = catalog.list_documents()
            for row in rows:
                if query.lower() in row["source_path"].lower() or query.lower() in row["title"].lower():
                    documents.append(row["source_path"])
        else:
            # query 없으면 "오늘" — dailynote/til 중 오늘 created_at인 것만
            today = date.today().isoformat()
            for dt in ("dailynote", "til"):
                for row in catalog.list_documents(doc_type=dt):
                    if row["created_at"].startswith(today):
                        documents.append(row["source_path"])

        return visualize_mindmap_img(documents)

    return [answer_question, write_daily, write_weekly, write_til, visualize_mindmap]
