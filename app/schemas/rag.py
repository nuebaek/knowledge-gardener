from typing import Literal

from pydantic import BaseModel


class TopicList(BaseModel):
    topics: list[str]
    umbrella: str = ""


class TurnResult(BaseModel):
    verdict: Literal["explained", "partial", "skip"]
    stay_on_topic: bool = False
    next_question: str | None = None


class ConverseRequest(BaseModel):
    message: str
    thread_id: str


class SavedDocument(BaseModel):
    type: str
    file_name: str


class ConverseResponse(BaseModel):
    answer: str
    tools_used: list[str]
    saved_documents: list[SavedDocument] = []
    mindmap_plaintext: str | None = None
    sources: list[str] = []
    recall: list[str] = []


class ThreadMessage(BaseModel):
    role: str
    content: str


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ThreadMessage]