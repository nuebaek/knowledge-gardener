from fastapi import APIRouter

from controllers import corpus as corpus_controller
from schemas import CorpusResponse

router = APIRouter()


@router.get("/corpus", response_model=CorpusResponse)
def corpus() -> CorpusResponse:
    return corpus_controller.get_corpus()
