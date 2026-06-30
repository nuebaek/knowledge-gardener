"""/ask 라우터 — 질문을 받아 RAG 답변을 반환."""
from fastapi import APIRouter, Request

from controllers import rag as rag_controller
from schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request) -> AskResponse:
    # 체인은 서버 시작 시 lifespan에서 1회 구성돼 app.state.rag에 보관됨
    rag_chain = request.app.state.rag
    return rag_controller.ask(rag_chain, req.question)
