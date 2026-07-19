from pydantic import BaseModel


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
