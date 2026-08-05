from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.rag.chain import answer_document_question, build_bm25, bm25_search, hybrid_merge, rerank_docs


def _doc(source, text):
    return Document(page_content=text, metadata={"source": source})


def test_bm25_search_ranks_keyword_match_first():
    docs = [
        _doc("a.md", "고양이는 포유류다"),
        _doc("b.md", "트랜스포머는 attention 기반 신경망이다"),
        _doc("c.md", "강아지도 포유류다"),
    ]
    bm25 = build_bm25(docs)
    hits = bm25_search(bm25, docs, "트랜스포머 attention", k=2)
    assert hits[0].metadata["source"] == "b.md"


def test_bm25_search_excludes_zero_score_docs():
    docs = [_doc("a.md", "고양이는 포유류다"), _doc("b.md", "강아지도 포유류다")]
    bm25 = build_bm25(docs)
    assert bm25_search(bm25, docs, "우주왕복선 발사 절차", k=2) == []


def test_hybrid_merge_dedupes_and_boosts_agreement():
    shared = _doc("a.md", "공유 문서")
    dense_only = _doc("b.md", "dense만 찾은 문서")
    bm25_only = _doc("c.md", "bm25만 찾은 문서")

    merged = hybrid_merge(dense_docs=[shared, dense_only], bm25_docs=[shared, bm25_only], k=3)

    sources = [d.metadata["source"] for d in merged]
    assert sources[0] == "a.md"  # 양쪽에서 다 나온 문서가 1위
    assert set(sources) == {"a.md", "b.md", "c.md"}  # 중복 없이 합집합


def test_hybrid_merge_respects_k():
    docs = [_doc(f"{i}.md", f"문서 {i}") for i in range(5)]
    merged = hybrid_merge(dense_docs=docs, bm25_docs=[], k=2)
    assert len(merged) == 2


def test_hybrid_merge_default_rrf_k_matches_formula():
    dense_docs = [_doc(f"d{i}.md", f"dense {i}") for i in range(3)]
    bm25_docs = [_doc(f"b{i}.md", f"bm25 {i}") for i in range(3)]

    merged = hybrid_merge(dense_docs=dense_docs, bm25_docs=bm25_docs, k=6)

    expected_scores = {}
    for source in (dense_docs, bm25_docs):
        for rank, doc in enumerate(source, 1):
            expected_scores[doc.metadata["source"]] = (
                expected_scores.get(doc.metadata["source"], 0.0) + 1 / (60 + rank)
            )
    expected_order = sorted(expected_scores, key=expected_scores.get, reverse=True)
    assert [d.metadata["source"] for d in merged] == expected_order


def test_hybrid_merge_rrf_k_reorders_results():
    # x: rank-1 in one list only. y: rank-5 in both lists (agreement, but low rank).
    # small rrf_k weights top rank heavily -> x wins. large rrf_k flattens rank decay,
    # so agreement (appearing twice) wins instead -> y overtakes x.
    x = _doc("x.md", "x")
    y = _doc("y.md", "y")
    dense_docs = [x, _doc("d1.md", "d1"), _doc("d2.md", "d2"), _doc("d3.md", "d3"), y]
    bm25_docs = [_doc("e1.md", "e1"), _doc("e2.md", "e2"), _doc("e3.md", "e3"), _doc("e4.md", "e4"), y]

    small = [d.metadata["source"] for d in hybrid_merge(dense_docs, bm25_docs, k=9, rrf_k=1)]
    large = [d.metadata["source"] for d in hybrid_merge(dense_docs, bm25_docs, k=9, rrf_k=1000)]

    assert small.index("x.md") < small.index("y.md")
    assert large.index("y.md") < large.index("x.md")


class _FakeReranker:
    """실제 cross-encoder 대신 (query, doc_text) 쌍 -> 점수 매핑을 그대로 돌려준다."""

    def __init__(self, score_by_text: dict):
        self.score_by_text = score_by_text
        self.received_pairs = None

    def predict(self, pairs):
        self.received_pairs = pairs
        return [self.score_by_text[text] for _, text in pairs]


def test_rerank_docs_dedupes_candidates_appearing_in_both_pools():
    shared = _doc("a.md", "공유 문서")
    dense_only = _doc("b.md", "dense만")
    reranker = _FakeReranker({"공유 문서": 0.5, "dense만": 0.1})

    docs, scores = rerank_docs(reranker, "질의", dense_docs=[shared, dense_only], bm25_docs=[shared], k=5)

    assert len(reranker.received_pairs) == 2  # 중복 제거 후에만 채점
    assert [d.metadata["source"] for d in docs] == ["a.md", "b.md"]
    assert scores == [0.5, 0.1]


def test_rerank_docs_orders_by_score_not_by_original_rank():
    low_first = _doc("low.md", "낮은 점수")
    high_second = _doc("high.md", "높은 점수")
    reranker = _FakeReranker({"낮은 점수": 0.1, "높은 점수": 0.9})

    docs, scores = rerank_docs(reranker, "질의", dense_docs=[low_first, high_second], bm25_docs=[], k=5)

    assert [d.metadata["source"] for d in docs] == ["high.md", "low.md"]
    assert scores == [0.9, 0.1]


def test_rerank_docs_respects_k():
    docs_in = [_doc(f"{i}.md", f"문서{i}") for i in range(5)]
    reranker = _FakeReranker({f"문서{i}": float(i) for i in range(5)})

    docs, scores = rerank_docs(reranker, "질의", dense_docs=docs_in, bm25_docs=[], k=2)

    assert len(docs) == 2
    assert len(scores) == 2


def test_rerank_docs_empty_pool_returns_empty():
    reranker = _FakeReranker({})
    docs, scores = rerank_docs(reranker, "질의", dense_docs=[], bm25_docs=[], k=5)
    assert docs == []
    assert scores == []


class _FakeChatModel:
    """BaseChatModel이 아니므로 apply_fallback이 그대로 통과시킨다 — 폴백 체인 없이 단순 호출만 검증."""

    def __init__(self, answer: str):
        self.answer = answer
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages.to_messages()
        return AIMessage(content=self.answer)

    def __call__(self, messages):
        return self.invoke(messages)


def test_answer_document_question_grounds_in_document_text(monkeypatch):
    from app.rag import chain

    fake = _FakeChatModel("문서에 따르면 X는 Y다.")
    monkeypatch.setattr(chain, "build_llm", lambda: fake)

    result = chain.answer_document_question("X가 뭐야?", "X는 Y라는 개념이다.", [])

    assert result == "문서에 따르면 X는 Y다."
    system_content = fake.received_messages[0].content
    human_content = fake.received_messages[1].content
    assert "X는 Y라는 개념이다." in system_content
    assert "X가 뭐야?" in human_content


def test_answer_document_question_includes_prior_history(monkeypatch):
    from app.rag import chain

    fake = _FakeChatModel("이어지는 답변")
    monkeypatch.setattr(chain, "build_llm", lambda: fake)

    history = [
        {"role": "user", "content": "첫 질문"},
        {"role": "assistant", "content": "첫 답변"},
    ]
    chain.answer_document_question("두번째 질문", "문서 내용", history)

    human_content = fake.received_messages[1].content
    assert "첫 질문" in human_content
    assert "첫 답변" in human_content
    assert "두번째 질문" in human_content
