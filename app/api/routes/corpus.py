from fastapi import APIRouter

from app.services import corpus_service
from app.schemas.corpus import CorpusResponse, DocumentDetail, DocumentSummary, SearchResponse

router = APIRouter()


@router.get("/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    return corpus_service.get_corpus()


@router.get("/documents", response_model=list[DocumentSummary])
def documents() -> list[DocumentSummary]:
    return corpus_service.list_documents()


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
def document(doc_id: str) -> DocumentDetail:
    return corpus_service.get_document(doc_id)


@router.get("/search", response_model=SearchResponse)
def search(q: str = "") -> SearchResponse:
    return corpus_service.search_documents(q)
