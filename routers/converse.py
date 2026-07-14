"""/converse 라우터 — agent 실행."""
from fastapi import APIRouter, Request

from controllers import rag as rag_controller
from schemas import ConverseRequest, ConverseResponse, ThreadHistoryResponse

router = APIRouter()

@router.post("/converse", response_model=ConverseResponse)
def converse(req: ConverseRequest, request: Request) -> ConverseResponse:
    agent_graph = request.app.state.agent
    return rag_controller.converse(agent_graph, req.message, req.thread_id)


@router.get("/threads/{thread_id}")
def get_thread(thread_id: str, request: Request) -> ThreadHistoryResponse:
    return rag_controller.get_thread_history(request.app.state.agent, thread_id)