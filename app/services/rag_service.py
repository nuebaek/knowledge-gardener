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


def converse(agent_graph, message: str, thread_id: str, selected_topics: list[str] | None = None) -> ConverseResponse:
    config = {"configurable": {"thread_id": thread_id}}
    before = agent_graph.get_state(config).values

    graph_input = {"messages": [("human", message)]}
    if selected_topics is not None:
        graph_input["selected_topics"] = selected_topics

    start = time.perf_counter()
    try:
        after = agent_graph.invoke(graph_input, config=config)
    except Exception as exc:
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
    # 이번 턴에 패스/부분설명으로 새로 seedling이 된 토픽 — apply_verdict가 seedlings에
    # 통째로 추가한 걸 그대로 쓰므로, before 이후로 늘어난 만큼만 diff로 잘라낸다.
    new_seedlings = _new_slice(before, after, "seedlings")

    return ConverseResponse(
        answer=_content_text(after["messages"][-1].content),
        tools_used=tools_used,
        saved_documents=saved_documents,
        mindmap_plaintext=mindmaps[-1] if mindmaps else None,
        sources=sources_calls[-1] if sources_calls else [],
        recall=[s["topic"] for s in new_seedlings],
        awaiting_topic_confirm=after.get("awaiting_topic_confirm", False),
        topics=after.get("pending", []) if after.get("awaiting_topic_confirm") else [],
    )


def get_thread_history(agent_graph, thread_id: str) -> ThreadHistoryResponse:
    config = {"configurable": {"thread_id": thread_id}}
    messages = agent_graph.get_state(config).values.get("messages", [])
    return ThreadHistoryResponse(
        thread_id=thread_id,
        messages=[ThreadMessage(role=m.type, content=_content_text(m.content)) for m in messages],
    )
