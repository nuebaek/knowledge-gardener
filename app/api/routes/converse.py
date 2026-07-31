from fastapi import APIRouter, Request

from app.services import rag_service
from app.schemas.rag import ConverseRequest, ConverseResponse, ThreadHistoryResponse

router = APIRouter()

@router.post("/converse", response_model=ConverseResponse)
def converse(req: ConverseRequest, request: Request) -> ConverseResponse:
    agent_graph = request.app.state.agent
    return rag_service.converse(agent_graph, req.message, req.thread_id)


@router.get("/threads/{thread_id}")
def get_thread(thread_id: str, request: Request) -> ThreadHistoryResponse:
    return rag_service.get_thread_history(request.app.state.agent, thread_id)