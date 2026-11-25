"""루브릭 판정(decide) 단위 테스트 — LLM 관찰값을 받아 코드가 내리는 결정만 검사한다."""
from app.rag.chain import MAX_FOLLOWUPS, PASS_DEPTH, decide
from app.schemas.rag import DimScore, RubricResult


def _r(depth=(2, 2, 0, 0), hedged=False, skipped=False, follow_up="왜?", next_q="다음 토픽 설명해줄래?"):
    d, m, a, c = depth
    return RubricResult(
        skipped=skipped, hedged=hedged,
        definition=DimScore(evidence="", score=d), mechanism=DimScore(evidence="", score=m),
        application=DimScore(evidence="", score=a), connection=DimScore(evidence="", score=c),
        follow_up=follow_up, next_topic_question=next_q,
    )


def test_defaults_match_design():
    assert PASS_DEPTH == 4 and MAX_FOLLOWUPS == 2


def test_deep_confident_explanation_passes_and_moves_on():
    t = decide(_r(depth=(2, 1, 1, 0)), followups=0, has_next=True)
    assert (t.verdict, t.passed, t.stay_on_topic, t.outcome) == ("explained", True, False, "explained")
    assert t.next_question == "다음 토픽 설명해줄래?"


def test_mechanism_zero_does_not_block_when_total_depth_passes():
    """정의·비교형 개념(401 vs 403)은 원리가 0이어도 합계로 통과 — "원리 필수" 조건을 뺀 근거."""
    assert decide(_r(depth=(2, 0, 1, 2)), followups=0, has_next=True).passed is True


def test_shallow_explanation_gets_follow_up_targeting_weak_dim():
    t = decide(_r(depth=(1, 0, 0, 0), follow_up="왜 그렇게 동작해?"), followups=0, has_next=True)
    assert t.stay_on_topic is True
    assert t.next_question == "왜 그렇게 동작해?"


def test_follow_up_cap_records_shallow_confident_answer_as_seedling():
    t = decide(_r(depth=(1, 0, 1, 0)), followups=MAX_FOLLOWUPS, has_next=True)
    assert t.stay_on_topic is False
    assert t.verdict == "explained"      # 태도 라벨은 그대로(평가 호환)
    assert t.outcome == "partial"        # 기록은 🌱
    assert t.weak == ["mechanism", "connection"]


def test_hedged_is_partial_regardless_of_depth():
    t = decide(_r(depth=(2, 2, 2, 2), hedged=True), followups=0, has_next=True)
    assert (t.verdict, t.passed, t.outcome) == ("partial", True, "partial")
    assert t.weak == []                  # 다 2점이면 약한 항목 없음


def test_skip_moves_on_without_follow_up():
    t = decide(_r(skipped=True), followups=0, has_next=True)
    assert (t.verdict, t.stay_on_topic, t.passed) == ("skip", False, False)


def test_last_topic_has_no_next_question_even_if_llm_wrote_one():
    """next_topic이 없으면 LLM이 질문을 지어내도 코드가 버린다."""
    assert decide(_r(), followups=0, has_next=False).next_question is None


def test_no_follow_up_text_means_no_stay():
    """LLM이 꼬리질문을 비워 보내면 빈 턴을 만들지 않고 넘어간다."""
    assert decide(_r(depth=(1, 0, 0, 0), follow_up=None), followups=0, has_next=True).stay_on_topic is False
