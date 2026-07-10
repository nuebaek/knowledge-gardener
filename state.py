from typing_extensions import TypedDict
from langchain_core.documents import Document

# State 정의
class GraphState(TypedDict):
    question: str
    document: list[Document]
    answer: str
    sources: list[str]