"""인출연습 세션 상태(StudySessionState)의 '기록' 로직 단위 테스트.

LLM이 하는 판단(explained/partial/skip)이 아니라, 그 판정을 상태에 올바르게 적재하는지만
검사한다 — 순수 함수라 LLM 호출도 fixture도 필요 없다.
"""
from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from app.rag.study_session import new_session, apply_verdict, is_complete


def test_new_session_puts_all_topics_in_pending():
    state = new_session(["레이어 캐시", "bind-mount"])
    assert state["pending"] == ["레이어 캐시", "bind-mount"]
    assert state["answered"] == []
    assert state["seedlings"] == []


def test_explained_topic_records_user_wording():
    """설명한 토픽은 사용자 표현 그대로 answered 에 남는다 (표현 업그레이드 금지 원칙)."""
    state = new_session(["레이어 캐시"])
    state = apply_verdict(state, "레이어 캐시", "explained", "변경 없는 층을 재사용해서 빌드가 빨라짐")

    assert state["answered"] == [
        {"topic": "레이어 캐시", "explanation": "변경 없는 층을 재사용해서 빌드가 빨라짐"}
    ]
    assert state["seedlings"] == []


def test_skipped_topic_becomes_seedling_not_dropped():
    """회귀: 스킵한 토픽이 조용히 사라지지 않고 seedlings 에 남아야 한다."""
    state = new_session(["레이어 캐시", "bind-mount"])
    state = apply_verdict(state, "레이어 캐시", "explained", "변경 없는 층 재사용")
    state = apply_verdict(state, "bind-mount", "skip", "모르겠어 넘어가")

    seedling_topics = [s["topic"] for s in state["seedlings"]]
    answered_topics = [a["topic"] for a in state["answered"]]

    assert "bind-mount" in seedling_topics          # 🌱 로 남았다
    assert "bind-mount" not in answered_topics       # answered 로 잘못 새지 않았다


def test_partial_attempt_goes_to_seedling_with_its_wording():
    """불확실한 시도(partial)는 answered 로 새지 않고 시도한 말 그대로 🌱 에 남는다."""
    state = new_session(["LoRA"])
    state = apply_verdict(state, "LoRA", "partial", "어댑터를 끼워서 일부만 학습하는 거... 맞나?")

    assert state["answered"] == []
    assert state["seedlings"] == [
        {"topic": "LoRA", "user_wording": "어댑터를 끼워서 일부만 학습하는 거... 맞나?"}
    ]


def test_addressed_topic_leaves_pending():
    """회귀: 한 번 다룬 토픽은 pending 에서 빠져 다시 안 물어봐야 한다."""
    state = new_session(["레이어 캐시", "bind-mount"])
    state = apply_verdict(state, "레이어 캐시", "explained", "...")

    assert "레이어 캐시" not in state["pending"]
    assert state["pending"] == ["bind-mount"]        # 안 다룬 것만 남는다


def test_session_not_complete_until_all_topics_addressed():
    state = new_session(["레이어 캐시", "bind-mount"])
    assert is_complete(state) is False

    state = apply_verdict(state, "레이어 캐시", "explained", "...")
    assert is_complete(state) is False               # bind-mount 아직 안 다룸

    state = apply_verdict(state, "bind-mount", "skip", "...")
    assert is_complete(state) is True                # 둘 다 다뤘으니 종료 가능


# --- study_node: LLM(MagicMock)이 낸 결과를 받아 상태에 적재하는 노드 로직 ------
from app.schemas.rag import TurnResult          # noqa: E402
from app.rag.nodes import FINALIZE_CONFIRM_MESSAGE, make_study_node  # noqa: E402


def _study_node_with(turn_result: TurnResult):
    """study_turn이 turn_result를 내도록 고정한 fake LLM으로 study_node를 만든다.
    gen_llm(토픽 추출용)은 이 테스트들이 안 건드리니 같은 mock을 재사용."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = turn_result
    study_node, _, _, _ = make_study_node(llm, llm, search_sources=lambda topic: [])
    return study_node


def _study_node_with_sequence(turn_results: list[TurnResult]):
    """호출마다 turn_results를 순서대로 내는 fake LLM으로 study_node를 만든다(다회 라운드 테스트용)."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = turn_results
    study_node, _, _, _ = make_study_node(llm, llm, search_sources=lambda topic: [])
    return study_node


def test_study_node_records_answer_and_asks_next_when_topics_remain():
    """남은 토픽이 있으면: 현재 답을 answered로 옮기고, LLM이 낸 다음 질문을 messages에 붙인다."""
    study_node = _study_node_with(
        TurnResult(verdict="explained", next_question="bind-mount는 네 말로 뭐야?")
    )
    state = {"messages": [HumanMessage("변경 없는 층 재사용")],
             "pending": ["레이어 캐시", "bind-mount"], "answered": [], "seedlings": []}

    out = study_node(state)

    assert out["pending"] == ["bind-mount"]                          # 현재 토픽 빠짐
    assert out["answered"] == [{"topic": "레이어 캐시", "explanation": "변경 없는 층 재사용"}]
    assert out["messages"][0].content == "bind-mount는 네 말로 뭐야?"  # 다음 질문 붙음


def test_study_node_asks_finalize_confirmation_on_last_topic():
    """마지막 토픽이면: 바로 저장하지 않고 "정리할까요?" 확인부터 묻는다(awaiting_finalize=True).
    (곧장 저장하면 사용자가 더 얘기하고 싶어도 끊겨버리는 문제가 된다.)"""
    study_node = _study_node_with(TurnResult(verdict="skip", next_question=None))
    state = {"messages": [HumanMessage("모르겠어")],
             "pending": ["레이어 캐시"], "answered": [], "seedlings": []}

    out = study_node(state)

    assert out["pending"] == []
    assert out["seedlings"] == [{"topic": "레이어 캐시", "user_wording": "모르겠어"}]
    assert out["awaiting_finalize"] is True
    assert out["messages"][0].content == FINALIZE_CONFIRM_MESSAGE
    assert is_complete(out) is True


def test_study_node_stays_on_topic_without_touching_pending_when_deepening():
    """stay_on_topic=True면: pending/answered/seedlings는 안 건드리고 버퍼(current_explanation)만 쌓는다.
    (건드리면 파고드는 도중에 토픽이 끝난 걸로 잘못 기록되는 버그가 된다.)"""
    study_node = _study_node_with(
        TurnResult(verdict="explained", stay_on_topic=True,
                   next_question="prefill이랑 decode는 뭐가 달라?")
    )
    state = {"messages": [HumanMessage("추론은 prefill과 decode로 나뉘어")],
             "pending": ["LLM Inference"], "answered": [], "seedlings": []}

    out = study_node(state)

    assert "pending" not in out and "answered" not in out and "seedlings" not in out
    assert out["current_explanation"] == "추론은 prefill과 decode로 나뉘어"
    assert out["messages"][0].content == "prefill이랑 decode는 뭐가 달라?"


def test_study_node_accumulates_explanation_across_deepen_then_resolve():
    """파고들기 라운드의 원문이 최종 기록에서 안 사라지고, 다음 라운드 답과 이어 붙어 기록된다
    (표현 그대로 원칙 — 첫 라운드 원문을 버리면 안 됨)."""
    study_node = _study_node_with_sequence([
        TurnResult(verdict="explained", stay_on_topic=True,
                   next_question="prefill이랑 decode는 뭐가 달라?"),
        TurnResult(verdict="explained", stay_on_topic=False, next_question=None),
    ])
    state = {"messages": [HumanMessage("추론은 prefill과 decode로 나뉘어")],
             "pending": ["LLM Inference"], "answered": [], "seedlings": []}

    round1 = study_node(state)
    state = {
        **state, **round1,
        "messages": state["messages"] + round1["messages"]
        + [HumanMessage("prefill은 kv cache 만들고, decode는 순차 생성")],
    }
    round2 = study_node(state)

    assert round2["pending"] == []
    assert round2["answered"] == [{
        "topic": "LLM Inference",
        "explanation": "추론은 prefill과 decode로 나뉘어\nprefill은 kv cache 만들고, decode는 순차 생성",
    }]
    assert round2["current_explanation"] == ""
