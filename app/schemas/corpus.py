from typing import Literal

from pydantic import BaseModel, Field


class CorpusResponse(BaseModel):
    count: int
    topics: list[str]


class DocumentSummary(BaseModel):
    id: str
    title: str
    excerpt: str
    char_count: int
    created_at: str
    doc_type: str
    tags: list[str] = []


class DocumentDetail(BaseModel):
    id: str
    title: str
    content: str
    char_count: int
    created_at: str
    doc_type: str
    tags: list[str] = []


class TagRequest(BaseModel):
    name: str = Field(min_length=1, max_length=30)


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


class DocChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class DocChatRequest(BaseModel):
    question: str
    history: list[DocChatMessage] = []


class DocChatResponse(BaseModel):
    answer: str
