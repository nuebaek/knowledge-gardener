from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="user query")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


class CorpusResponse(BaseModel):
    count: int
    topics: list[str]


class DocumentSummary(BaseModel):
    id: str
    title: str
    excerpt: str
    char_count: int


class DocumentDetail(BaseModel):
    id: str
    title: str
    content: str
    char_count: int


class SearchHit(BaseModel):
    doc_id: str
    title: str
    section: str | None = None
    snippet: str
    match_start: int
    match_end: int


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


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