from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="user query")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class CorpusResponse(BaseModel):
    count: int
    topics: list[str]


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


class ThreadMessage(BaseModel):
    role: str
    content: str


class ThreadHistoryResponse(BaseModel):
    thread_id: str
    messages: list[ThreadMessage]