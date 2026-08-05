import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest  # noqa: E402

from app.rag.chain import section_key, section_path  # noqa: E402
from benchmark import _dedupe, _ndcg, score_one  # noqa: E402
from gen_evalset import content_words, header_overlap, lang_of, public_sections  # noqa: E402


# ---- gold 라벨 왕복 ---------------------------------------------------------
# 벤치마크 전체가 여기에 걸려 있다. 생성기는 section_key를 doc/section으로 쪼개 JSONL에 쓰고,
# 러너는 그걸 다시 이어붙여 검색 결과와 비교한다. 두 포맷이 갈리면 모든 문항이 조용히 MISS가
# 되고, 검색기가 멀쩡한데 점수가 0인 상태를 디버깅하게 된다.

def _roundtrip(meta: dict) -> str:
    key = section_key(meta)
    doc, heading = key.split("#", 1)      # 생성기가 JSONL에 저장하는 방식
    return f"{doc}#{heading}"             # 러너가 gold를 복원하는 방식


def test_section_key_roundtrip():
    meta = {"source": "data/processed/rnn.md", "h1": "RNN", "h2": "LSTM"}
    assert _roundtrip(meta) == section_key(meta) == "data/processed/rnn.md#RNN > LSTM"


def test_section_key_roundtrip_survives_hash_in_heading():
    # cs231n 노트엔 'C#' 'f#(x)' 같은 헤더가 나올 수 있다. split("#", 1)이라 살아남아야 한다.
    meta = {"source": "data/processed/x.md", "h1": "Notes", "h2": "gradient of f#(x)"}
    assert _roundtrip(meta) == "data/processed/x.md#Notes > gradient of f#(x)"


def test_section_path_skips_missing_levels():
    # h2가 없는 섹션이 h1 > h3로 이어붙어야지, 빈 칸이 끼면 라벨이 달라진다
    assert section_path({"h1": "A", "h3": "C"}) == "A > C"
    assert section_path({}) == ""


# ---- 지표 -------------------------------------------------------------------

def test_dedupe_keeps_first_occurrence_order():
    # 한 섹션에서 청크가 3개 걸려도 순위 1개로 세야 MRR이 부풀지 않는다
    assert _dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_score_one_single_gold_at_rank_two():
    m = score_one(["x", "gold", "y"], {"gold"}, k=3)
    assert m["hit"] == 1.0
    assert m["recall"] == 1.0
    assert m["mrr"] == 0.5
    assert m["ndcg"] == pytest.approx(1 / 1.584962, rel=1e-4)


def test_score_one_miss_is_all_zero():
    assert score_one(["x", "y"], {"gold"}, k=3) == {"hit": 0.0, "recall": 0.0, "mrr": 0.0,
                                                    "ndcg": 0.0}


def test_score_one_partial_recall_on_multi_doc():
    m = score_one(["a", "x", "y"], {"a", "b"}, k=3)
    assert m["hit"] == 1.0 and m["recall"] == 0.5 and m["mrr"] == 1.0


def test_ndcg_rewards_earlier_ranks():
    assert _ndcg(["gold", "x"], {"gold"}, 2) > _ndcg(["x", "gold"], {"gold"}, 2)


def test_score_one_respects_k_cutoff():
    # k 밖에 있는 정답은 못 찾은 것으로 세야 한다
    assert score_one(["x", "y", "gold"], {"gold"}, k=2)["hit"] == 0.0


# ---- conceptual 문항의 헤더 어휘 회피 ----------------------------------------
# eval_retrieval.py 시절의 걱정("제목 패러프레이즈 매칭만 재는 것 아니냐")을 LLM 없이 막는 필터.

RNN_KEY = "data/processed/rnn.md#Recurrent Neural Networks > LSTM"


def test_header_overlap_flags_title_paraphrase():
    assert header_overlap("What is an LSTM in recurrent neural networks?", RNN_KEY) == 1.0


def test_header_overlap_passes_genuine_concept_question():
    q = "Why do plain sequence models forget information from many steps earlier, and what fixes it?"
    assert header_overlap(q, RNN_KEY) == 0.0


def test_header_overlap_strips_korean_particles():
    # 조사째로 비교하면 "게이트가" != "게이트"라 필터가 뚫린다. 조사가 붙든 안 붙든 같은 점수여야 함.
    key = "data/processed/week-05-TIL.md#게이트 구조"
    assert header_overlap("게이트 구조 설명해줘", key) == 1.0
    assert header_overlap("게이트가 구조를 어떻게 바꿔?", key) == 1.0


def test_header_overlap_is_partial_when_question_covers_some_of_heading():
    # 헤더가 3어절인데 질문이 2개만 덮으면 2/3. 임계값(0.4)을 넘으므로 여전히 탈락 대상.
    key = "data/processed/week-05-TIL.md#Transformer > 게이트 구조"
    assert header_overlap("게이트가 구조를 어떻게 바꿔?", key) == pytest.approx(2 / 3)


def test_content_words_drops_stopwords_and_single_chars():
    assert content_words("the LSTM of a cell") == {"lstm", "cell"}


# ---- 언어 판정 --------------------------------------------------------------

def test_lang_of():
    assert lang_of("A convolutional layer applies a filter across the image.") == "en"
    assert lang_of("합성곱 층은 필터를 이미지 전체에 걸쳐 적용한다.") == "ko"
    # 한국어 문서에 영어 용어가 많이 섞여도 ko여야 한다 (TIL이 전부 이런 형태)
    assert lang_of("LSTM은 forget gate로 cell state를 유지한다. gradient vanishing 완화.") == "ko"


# ---- 실제 코퍼스 스모크 ------------------------------------------------------

def test_public_sections_cover_both_languages():
    """공개 코퍼스가 실제로 두 언어 슬라이스를 만들어내는지. 한국어가 0이면 다국어 임베딩
    비교 자체가 성립하지 않는다 — 이게 원래 벤치마크를 새로 만든 이유다."""
    sections = public_sections()
    assert sections, "data/processed/에서 섹션이 하나도 안 나왔다"
    langs = {lang_of(body) for body in sections.values()}
    assert langs == {"en", "ko"}
    assert all("#" in key for key in sections)
    # 공개 코퍼스 밖(개인 노트)이 절대 섞이면 안 된다
    assert all(key.startswith("data/processed/") for key in sections)
