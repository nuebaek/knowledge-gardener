from fastapi import APIRouter

from controllers import corpus as corpus_controller
from schemas import CorpusResponse, DocumentDetail, DocumentSummary, SearchResponse

router = APIRouter()


@router.get("/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    return corpus_controller.get_corpus()


@router.get("/documents", response_model=list[DocumentSummary])
def documents() -> list[DocumentSummary]:
    return corpus_controller.list_documents()


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
def document(doc_id: str) -> DocumentDetail:
    return corpus_controller.get_document(doc_id)


@router.get("/search", response_model=SearchResponse)
def search(q: str = "") -> SearchResponse:
    return corpus_controller.search_documents(q)
