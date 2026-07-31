from fastapi import APIRouter

from app.services import corpus_service
from app.schemas.corpus import CorpusResponse, DocumentDetail, DocumentSummary, SearchResponse, TagRequest

router = APIRouter()


@router.get("/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    return corpus_service.get_corpus()


@router.get("/documents", response_model=list[DocumentSummary])
def documents(doc_type: str | None = None, tag: str | None = None) -> list[DocumentSummary]:
    return corpus_service.list_documents(doc_type=doc_type, tag=tag)


@router.get("/tags", response_model=list[str])
def tags() -> list[str]:
    return corpus_service.list_tags()


@router.get("/documents/{doc_id:path}", response_model=DocumentDetail)
def document(doc_id: str) -> DocumentDetail:
    return corpus_service.get_document(doc_id)


@router.post("/documents/{doc_id:path}/tags", response_model=list[str])
def add_tag(doc_id: str, body: TagRequest) -> list[str]:
    return corpus_service.add_tag(doc_id, body.name)


@router.delete("/documents/{doc_id:path}/tags/{name}", response_model=list[str])
def remove_tag(doc_id: str, name: str) -> list[str]:
    return corpus_service.remove_tag(doc_id, name)


@router.get("/search", response_model=SearchResponse)
def search(q: str = "") -> SearchResponse:
    return corpus_service.search_documents(q)
