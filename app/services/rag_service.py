"""RAG 컨트롤러 — 체인 호출과 응답 변환을 담당."""
from fastapi import HTTPException
from pathlib import Path
from app.schemas.rag import AskResponse, ConverseResponse, SavedDocument, ThreadHistoryResponse, ThreadMessage
from langchain.messages import AIMessage, ToolMessage


def ask(rag_chain, question: str) -> AskResponse:
    """체인을 호출해 답변과 출처를 반환. 실패 시 500으로 변환.

    관련 문서를 못 찾은 경우도 LLM이 "자료에 없다"고 정상적으로 답한 것이므로 200으로 반환한다.
    빈 sources가 바로 그 신호이며, 호출 측(프론트엔드)이 이를 보고 UI를 다르게 처리할 수 있다.
    """
    try:
        result = rag_chain.invoke({"question": question})
    except Exception as exc:  # LLM 호출 실패 등
        raise HTTPException(status_code=500, detail=f"RAG 실행 실패: {exc}") from exc

    return AskResponse(answer=result["answer"], sources=result.get("sources", []))


def converse(agent_graph, message: str, thread_id: str) -> ConverseResponse:
    """에이전트를 호출해 답변·사용된 tool·저장된 문서를 반환. 실패 시 500으로 변환.

    thread_id 기준으로 checkpointer가 대화 기록을 들고 있으므로, invoke 전 시점의
    메시지 개수(before_count)를 기준 삼아 이번 턴에 새로 생긴 메시지만 골라낸다.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent_graph.get_state(config)
    before_count = len(snapshot.values.get("messages", []))
    tools_used = []
    saved_documents = []

    try:
        result = agent_graph.invoke({"messages": [("human", message)]}, config=config)
        new_msg = result["messages"][before_count:]
        for m in new_msg:
            if isinstance(m, AIMessage) and m.tool_calls:
                tools_used.append(m.tool_calls[0]["name"])
            if isinstance(m, ToolMessage):
                doc_type = m.name
                file_name = Path(m.content).name
                saved_documents.append(SavedDocument(type=doc_type, file_name=file_name))

    except Exception as exc:  # LLM 호출 실패 등
        raise HTTPException(status_code=500, detail=f"Agent 실행 실패: {exc}") from exc

    return ConverseResponse(answer=result["messages"][-1].content, tools_used=tools_used, saved_documents=saved_documents)


def get_thread_history(agent_graph, thread_id: str) -> ThreadHistoryResponse:
    """디버그용: MemorySaver에 쌓인 상태를 필터링 없이 그대로 노출한다."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent_graph.get_state(config)
    messages = snapshot.values.get("messages", [])
    thread_messages = [ThreadMessage(role=m.type, content=m.content) for m in messages]
    return ThreadHistoryResponse(thread_id=thread_id, messages=thread_messages)