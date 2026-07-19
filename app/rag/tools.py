from langchain_core.tools import tool
from app.rag.graph import build_rag_graph
from app.writer.writer import write_daily_note, write_weekly_note, write_tilnote

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
    def write_daily(topic: str, learned: str, related_concepts: list[str] | None = None) -> str:
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
            learned: What was learned, described in the user's own words.
            related_concepts: Specific terms or concepts the user explicitly named.
                Leave empty if none were mentioned — do not add concepts on your own.
        """
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
        keywords: list[str],
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
            reflection: The user's reflection, expanded into full sentences if it was terse.
            actionplan: The next action, as stated by the user. Empty string if none mentioned.
            keywords: Short keywords, only from terms the user actually used.
        """
        return write_tilnote(what, learned, troubleshooting, reflection, actionplan, keywords)

    return [answer_question, write_daily, write_weekly, write_til]
