"""인출연습 세션 상태(StudySessionState)의 '기록' 로직 단위 테스트.

LLM이 하는 판단(explained/partial/skip)이 아니라, 그 판정을 상태에 올바르게 적재하는지만
검사한다 — 순수 함수라 LLM 호출도 fixture도 필요 없다.
"""
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

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
from app.schemas.rag import DimScore, RubricResult  # noqa: E402
from app.rag.nodes import FINALIZE_CONFIRM_MESSAGE, make_study_node  # noqa: E402


def _rubric(depth=(2, 2, 0, 0), hedged=False, skipped=False, follow_up=None, next_topic_question=None):
    """judge LLM의 관찰 결과. depth=(정의, 원리, 적용, 연결). 기본값은 확신 있고 깊이 4(통과)."""
    d, m, a, c = (0, 0, 0, 0) if skipped else depth
    return RubricResult(
        skipped=skipped, hedged=hedged,
        definition=DimScore(evidence="", score=d), mechanism=DimScore(evidence="", score=m),
        application=DimScore(evidence="", score=a), connection=DimScore(evidence="", score=c),
        follow_up=follow_up, next_topic_question=next_topic_question,
    )


def _study_node_with(rubric: RubricResult):
    """judge가 rubric을 내도록 고정한 fake LLM으로 study_node를 만든다.
    gen_llm(토픽 추출용)은 이 테스트들이 안 건드리니 같은 mock을 재사용."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = rubric
    study_node, _, _, _ = make_study_node(llm, llm, search_sources=lambda topic: [])
    return study_node


def _study_node_with_sequence(rubrics: list[RubricResult]):
    """호출마다 rubrics를 순서대로 내는 fake LLM으로 study_node를 만든다(다회 라운드 테스트용)."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.side_effect = rubrics
    study_node, _, _, _ = make_study_node(llm, llm, search_sources=lambda topic: [])
    return study_node


def test_study_node_records_answer_and_asks_next_when_topics_remain():
    """남은 토픽이 있으면: 현재 답을 answered로 옮기고, LLM이 낸 다음 질문을 messages에 붙인다."""
    study_node = _study_node_with(_rubric(next_topic_question="bind-mount는 네 말로 뭐야?"))
    state = {"messages": [HumanMessage("변경 없는 층 재사용")],
             "pending": ["레이어 캐시", "bind-mount"], "answered": [], "seedlings": []}

    out = study_node(state)

    assert out["pending"] == ["bind-mount"]                          # 현재 토픽 빠짐
    assert out["answered"] == [{"topic": "레이어 캐시", "explanation": "변경 없는 층 재사용"}]
    assert out["messages"][0].content == "bind-mount는 네 말로 뭐야?"  # 다음 질문 붙음


def test_study_node_asks_finalize_confirmation_on_last_topic():
    """마지막 토픽이면: 바로 저장하지 않고 "정리할까요?" 확인부터 묻는다(awaiting_finalize=True).
    (곧장 저장하면 사용자가 더 얘기하고 싶어도 끊겨버리는 문제가 된다.)"""
    study_node = _study_node_with(_rubric(skipped=True))
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
        _rubric(depth=(1, 1, 0, 0), follow_up="prefill이랑 decode는 뭐가 달라?")  # 깊이 2 < 4 → 꼬리질문
    )
    state = {"messages": [HumanMessage("추론은 prefill과 decode로 나뉘어")],
             "pending": ["LLM Inference"], "answered": [], "seedlings": []}

    out = study_node(state)

    assert "pending" not in out and "answered" not in out and "seedlings" not in out
    assert out["current_explanation"] == "추론은 prefill과 decode로 나뉘어"
    assert out["messages"][0].content == "prefill이랑 decode는 뭐가 달라?"
    assert out["followups"] == 1


def test_study_node_accumulates_explanation_across_deepen_then_resolve():
    """파고들기 라운드의 원문이 최종 기록에서 안 사라지고, 다음 라운드 답과 이어 붙어 기록된다
    (표현 그대로 원칙 — 첫 라운드 원문을 버리면 안 됨)."""
    study_node = _study_node_with_sequence([
        _rubric(depth=(1, 1, 0, 0), follow_up="prefill이랑 decode는 뭐가 달라?"),
        _rubric(depth=(2, 2, 0, 1)),
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
    assert round2["followups"] == 0  # 다음 토픽은 상한을 새로 센다


def test_study_node_stops_following_up_at_cap_and_records_weak_dims():
    """꼬리질문 상한(2회)에 닿으면 더 묻지 않고 🌱로 넘기며, 약한 항목을 함께 남긴다
    (상한이 없으면 judge가 계속 얕다고 할 때 같은 토픽에서 무한 질문)."""
    study_node = _study_node_with(_rubric(depth=(1, 0, 1, 0), follow_up="왜 그렇게 동작해?"))
    state = {"messages": [HumanMessage("세 번째 시도")], "pending": ["Docker"],
             "answered": [], "seedlings": [], "current_explanation": "첫 시도\n두 번째 시도", "followups": 2}

    out = study_node(state)

    assert out["pending"] == []
    assert out["seedlings"] == [{
        "topic": "Docker",
        "user_wording": "첫 시도\n두 번째 시도\n세 번째 시도",
        "weak": ["mechanism", "connection"],
    }]
    assert out["followups"] == 0


def test_confident_but_hedged_deep_explanation_still_goes_to_seedling():
    """깊이는 통과해도 스스로 얼버무렸으면(hedged) answered가 아니라 🌱 — 확신 없는 이해는 다시 꺼낸다."""
    study_node = _study_node_with(_rubric(depth=(2, 2, 1, 1), hedged=True))
    state = {"messages": [HumanMessage("음.. 아마 이런 거 같은데")], "pending": ["LoRA"],
             "answered": [], "seedlings": []}

    out = study_node(state)

    assert out["answered"] == []
    assert out["seedlings"][0]["topic"] == "LoRA"


# --- confirm_finalize_node: "정리해서 저장할까요?"에 대한 답 처리 -----------------
from app.schemas.rag import TopicList                                    # noqa: E402


def _confirm_finalize_node_with(extracted: TopicList, write_daily_note):
    """토픽 추출기가 extracted를 내도록 고정한 fake LLM으로 confirm_finalize_node를 만든다."""
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = extracted
    _, _, confirm_finalize_node, _ = make_study_node(llm, llm, search_sources=lambda topic: [])
    return confirm_finalize_node


def _finished_session_state(reply: str):
    """세 토픽을 다 다루고 awaiting_finalize 상태에서 reply를 받은 상태."""
    return {
        "messages": [
            HumanMessage("오늘 Transformer 공부했어. self attention, positional encoding, multi head attention"),
            AIMessage(FINALIZE_CONFIRM_MESSAGE),
            HumanMessage(reply),
        ],
        "pending": [],
        "answered": [{"topic": "self attention", "explanation": "Query/Key/Value로 관련도를 계산해"}],
        "seedlings": [
            {"topic": "positional encoding", "user_wording": "모르겠어"},
            {"topic": "multi head attention", "user_wording": "잘 모르겠어"},
        ],
        "umbrella": "Transformer",
    }


def test_confirm_finalize_saves_when_user_says_yes(tmp_path, monkeypatch):
    """회귀: "응 저장해줘"에 저장이 실행돼야 한다.

    대화 전체를 토픽 추출기에 넣던 시절에는 세션을 시작시킨 발화가 매번 재추출돼서
    topics가 비지 않았고, 저장 대신 이미 끝낸 첫 토픽(self attention)을 다시 물었다.
    """
    saved = []
    monkeypatch.setattr("app.rag.nodes.save_raw_session", lambda **kw: tmp_path / "raw.md")
    monkeypatch.setattr("app.rag.nodes.write_daily_note",
                        lambda **kw: saved.append(kw) or (tmp_path / "transformer.md"))

    # 추출기가 "응 저장해줘"에서 아무 주제도 못 찾는 정상 케이스
    llm = MagicMock()
    extractor = llm.with_structured_output.return_value
    extractor.invoke.return_value = TopicList(topics=[], umbrella="")
    _, _, node, _ = make_study_node(llm, llm, search_sources=lambda topic: [])

    out = node(_finished_session_state("응 저장해줘"))

    assert out["awaiting_finalize"] is False
    assert out["saved_documents"] == [{"type": "dailynote", "file_name": "transformer.md"}]
    assert "저장했어요" in out["messages"][0].content
    assert "pending" not in out                      # 새 세션을 열지 않는다

    # 추출기에 들어간 건 사용자의 답 한 줄뿐 — 세션을 시작시킨 발화가 섞이면 안 된다.
    prompt_text = str(extractor.invoke.call_args.args[0])
    assert "응 저장해줘" in prompt_text
    assert "오늘 Transformer 공부했어" not in prompt_text


def test_confirm_finalize_ignores_already_covered_topics(tmp_path, monkeypatch):
    """추출기가 이미 다룬 토픽을 뱉어도(짧은 답변에서 흔함) 재질문하지 않고 저장한다."""
    monkeypatch.setattr("app.rag.nodes.save_raw_session", lambda **kw: tmp_path / "raw.md")
    monkeypatch.setattr("app.rag.nodes.write_daily_note", lambda **kw: tmp_path / "transformer.md")

    node = _confirm_finalize_node_with(
        TopicList(topics=["self attention", "positional encoding"], umbrella="Transformer"), None)
    out = node(_finished_session_state("응 저장해줘"))

    assert out["saved_documents"] == [{"type": "dailynote", "file_name": "transformer.md"}]
    assert "pending" not in out


def test_confirm_finalize_continues_session_on_new_topic(tmp_path, monkeypatch):
    """"더 이야기하고 싶은 게 있으면" 경로: 새 주제를 말하면 저장 대신 세션을 이어간다."""
    monkeypatch.setattr("app.rag.nodes.generate_recall_question",
                        lambda llm, topic, conversation: f"{topic} 네 말로 설명해볼래?")

    node = _confirm_finalize_node_with(
        TopicList(topics=["residual connection"], umbrella="Transformer"), None)
    out = node(_finished_session_state("아 맞다 residual connection도 배웠어"))

    assert out["pending"] == ["residual connection"]
    assert out["awaiting_finalize"] is False
    assert out["messages"][0].content == "residual connection 네 말로 설명해볼래?"
    assert "saved_documents" not in out


def test_daily_note_shows_weak_dims_of_seedling_in_korean():
    from app.rag.study_session import serialize_for_daily_note

    state = new_session(["Docker"])
    state = apply_verdict(state, "Docker", "partial", "이미지로 배포", weak=["mechanism", "connection"])

    note = serialize_for_daily_note(state)
    assert "- Docker: 이미지로 배포" in note
    assert "약한 부분: 원리, 연결" in note
