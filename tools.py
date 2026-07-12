from langchain_core.tools import tool
from graph import build_rag_graph


def make_tools():
    qa_graph = build_rag_graph()

    @tool
    def answer_question(question: str) -> str:
        """문서 코퍼스를 검색해 질문에 답한다.
        사용자가 개념을 묻거나, 설명을 요청하거나, "~가 뭐야", "~는 어떻게 동작해" 같은 질문을 할 때 사용한다.
        오늘 공부한 내용을 기록하려는 요청에는 사용하지 않는다."""
        result = qa_graph.invoke({"question": question})
        sources = ", ".join(result.get("sources", [])) or "없음"
        return f"{result['answer']}\n\n(출처: {sources})"

    return [answer_question]

