from typing import Annotated

from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import MessagesState

class GraphState(TypedDict):
    question: str
    document: list[Document]
    answer: str
    sources: list[str]
    doc_scores: list[float]
    is_relevant: bool
    retry_count: int
    rewritten_question: str | None


def append_records(left, right):
    return (left or []) + (right or [])


class StudySessionState(MessagesState):
    pending: list[str]
    answered: list[dict]
    seedlings: list[dict]
    umbrella: str
    current_explanation: str
    awaiting_finalize: bool
    saved_documents: Annotated[list[dict], append_records]
    mindmaps: Annotated[list[str], append_records]
    sources: Annotated[list[list[str]], append_records]
