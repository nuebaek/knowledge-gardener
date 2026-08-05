import logging

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, SystemMessage
from app.rag.chain import (
    _extract_sources, apply_fallback, TOP_K, RELEVANCE_THRESHOLD, study_turn,
    TOPIC_EXTRACT_PROMPT, generate_recall_question,
)
from app.rag.state import GraphState, StudySessionState
from app.rag.study_session import apply_verdict, flatten_conversation, is_complete, serialize_for_daily_note
from app.schemas.rag import TopicList
from app.writer.writer import save_raw_session, write_daily_note

logger = logging.getLogger(__name__)

MAX_CONTEXT_MESSAGES = 20

FINALIZE_CONFIRM_MESSAGE = "수고했어요! 오늘 배운 내용 정리해서 저장할까요? 더 이야기하고 싶은 게 있으면 말해주세요."


def make_agent_node(llm, tools, system_prompt):
    llm_with_tools = apply_fallback(llm, lambda m: m.bind_tools(tools))

    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"][-MAX_CONTEXT_MESSAGES:]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node


def make_nodes(vectorstore, prompt, rewrite_prompt, llm):
    llm = apply_fallback(llm)

    def retrieve_node(state: GraphState):
        rewritten_question = state.get("rewritten_question", "")
        if rewritten_question:
            question = rewritten_question
        else:
            question = state["question"]
        results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
        return {
            "document": [doc for doc, _ in results],
            "doc_scores": [score for _, score in results],
        }

    def generate_node(state: GraphState):
        context = "\n\n".join(d.page_content for d in state["document"])
        messages = prompt.invoke({"context": context, "question": state["question"]}).to_messages()
        response = llm.invoke(messages)
        sources = _extract_sources(state["document"]) if state.get("is_relevant", True) else []
        return {
            "answer": response.content,
            "sources": sources,
        }

    def grade_docs_node(state: GraphState):
        top_score = max(state["doc_scores"])
        is_relevant = top_score > RELEVANCE_THRESHOLD
        logger.debug("grade_docs top_score=%.3f threshold=%s -> is_relevant=%s", top_score, RELEVANCE_THRESHOLD, is_relevant)
        return {"is_relevant": is_relevant}


    def rewrite_query_node(state: GraphState):
        current_query = state.get("rewritten_question") or state["question"]
        messages = rewrite_prompt.invoke({
            "question": state["question"],
            "rewritten_question": state.get("rewritten_question", ""),
        }).to_messages()
        response = llm.invoke(messages)
        logger.debug("rewrite_query attempt %d: %r -> %r", state.get('retry_count', 0) + 1, current_query, response.content)
        return {
            "rewritten_question": response.content,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    return retrieve_node, generate_node, grade_docs_node, rewrite_query_node


def make_study_node(judge_llm, gen_llm, search_sources):
    topic_extractor = apply_fallback(gen_llm, lambda m: m.with_structured_output(TopicList))

    def study_node(state: StudySessionState):
        last_user_msg = state["messages"][-1].content
        pending = state["pending"]
        current_topic = pending[0]
        next_topic = pending[1] if len(pending) > 1 else None
        conversation = flatten_conversation(state["messages"])
        prior_explanation = state.get("current_explanation", "")

        result = study_turn(judge_llm, current_topic, next_topic, last_user_msg, conversation)
        logger.debug(
            "study_turn topic=%r next_topic=%r -> verdict=%s stay_on_topic=%s next_question=%r",
            current_topic, next_topic, result.verdict, result.stay_on_topic, result.next_question,
        )

        full_explanation = f"{prior_explanation}\n{last_user_msg}" if prior_explanation else last_user_msg

        if result.stay_on_topic:
            out = {"current_explanation": full_explanation}
            if result.next_question:
                out["messages"] = [AIMessage(result.next_question)]
            return out

        new_state = apply_verdict(state, current_topic, result.verdict, full_explanation)
        out = {
            "pending": new_state["pending"],
            "answered": new_state["answered"],
            "seedlings": new_state["seedlings"],
            "current_explanation": "",
        }
        if not is_complete(new_state):
            if result.next_question:
                out["messages"] = [AIMessage(result.next_question)]
            return out

        out["awaiting_finalize"] = True
        out["messages"] = [AIMessage(FINALIZE_CONFIRM_MESSAGE)]
        return out

    def finalize_node(state: StudySessionState):
        enriched_seedlings = []
        for s in state["seedlings"]:
            try:
                paths = search_sources(s["topic"])
            except Exception:
                logger.exception("seedling 근거 검색 실패: %s", s["topic"])
                paths = []
            enriched_seedlings.append({**s, "source_paths": paths})

        topics = [a["topic"] for a in state["answered"]] + [s["topic"] for s in state["seedlings"]]
        payload = {
            "topic": state.get("umbrella") or ", ".join(topics),
            "learned": serialize_for_daily_note(
                {"answered": state["answered"], "seedlings": enriched_seedlings}
            ),
            "related_concepts": topics,
        }
        raw_path = save_raw_session(**payload)

        try:
            path = write_daily_note(**payload)
            return {
                "awaiting_finalize": False,
                "saved_documents": [{"type": "dailynote", "file_name": path.name}],
                "messages": [AIMessage(f"오늘 학습 정리 끝, {path.name}으로 저장했어요")],
            }
        except Exception:
            logger.exception("write_daily_note 실패 — 원본은 %s에 보존됨", raw_path.name)

        return {
            "awaiting_finalize": False,
            "saved_documents": [{"type": "dailynote_raw", "file_name": raw_path.name}],
            "messages": [AIMessage(
                f"오늘 답변은 저장했는데({raw_path.name}) 노트 정리에는 실패했어요. "
                "잠시 후 다시 시도해주세요"
            )],
        }

    def confirm_finalize_node(state: StudySessionState):
        reply = state["messages"][-1].content
        messages = TOPIC_EXTRACT_PROMPT.invoke({"conversation": reply}).to_messages()
        topics = topic_extractor.invoke(messages)

        if not topics.topics:
            return finalize_node(state)

        conversation = flatten_conversation(state["messages"])
        question = generate_recall_question(gen_llm, topics.topics[0], conversation)
        return {
            "pending": topics.topics,
            "umbrella": state.get("umbrella") or topics.umbrella,
            "awaiting_finalize": False,
            "messages": [AIMessage(question)],
        }

    return study_node, finalize_node, confirm_finalize_node, confirm_topics_node
