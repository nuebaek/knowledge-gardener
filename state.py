from typing_extensions import TypedDict
from langchain_core.documents import Document

# State 정의
class GraphState(TypedDict):
    question: str
    document: list[Document]
    answer: str
    sources: list[str]
    doc_scores: list[float]
    is_relevant: bool
    retry_count: int
    rewritten_question: str | None
