import logging
import time

from fastapi import HTTPException
from langchain_core.messages import AIMessage

from app.schemas.rag import ConverseResponse, SavedDocument, ThreadHistoryResponse, ThreadMessage

logger = logging.getLogger(__name__)


def _new_slice(before: dict, after: dict, key: str) -> list:
    return after.get(key, [])[len(before.get(key, [])):]


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)


def converse(agent_graph, message: str, thread_id: str) -> ConverseResponse:
    config = {"configurable": {"thread_id": thread_id}}
    before = agent_graph.get_state(config).values

    start = time.perf_counter()
    try:
        after = agent_graph.invoke({"messages": [("human", message)]}, config=config)
    except Exception as exc:  # LLM 호출 실패 등
        raise HTTPException(status_code=500, detail=f"Agent 실행 실패: {exc}") from exc
    finally:
        logger.info("converse thread_id=%s elapsed_ms=%.0f", thread_id, (time.perf_counter() - start) * 1000)

    tools_used = [
        call["name"]
        for m in _new_slice(before, after, "messages")
        if isinstance(m, AIMessage)
        for call in (m.tool_calls or [])
    ]
    saved_documents = [SavedDocument(**doc) for doc in _new_slice(before, after, "saved_documents")]
    mindmaps = _new_slice(before, after, "mindmaps")
    sources_calls = _new_slice(before, after, "sources")

    return ConverseResponse(
        answer=_content_text(after["messages"][-1].content),
        tools_used=tools_used,
        saved_documents=saved_documents,
        mindmap_plaintext=mindmaps[-1] if mindmaps else None,
        sources=sources_calls[-1] if sources_calls else [],
    )


def get_thread_history(agent_graph, thread_id: str) -> ThreadHistoryResponse:
    config = {"configurable": {"thread_id": thread_id}}
    messages = agent_graph.get_state(config).values.get("messages", [])
    return ThreadHistoryResponse(
        thread_id=thread_id,
        messages=[ThreadMessage(role=m.type, content=_content_text(m.content)) for m in messages],
    )
