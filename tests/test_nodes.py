from langchain_core.documents import Document

from app.rag.chain import RELEVANCE_THRESHOLD, build_bm25
from app.rag.nodes import make_nodes


def _grade_docs_node():
    _, _, grade_docs_node, _ = make_nodes(None, None, None, None)
    return grade_docs_node


def _doc(score: float) -> Document:
    return Document(page_content=f"score={score}", metadata={"source": f"doc-{score}"})


class _FakeVectorstore:
    def __init__(self, docs, score=0.9):
        self.docs = docs
        self.score = score

    def similarity_search(self, query, k):
        return self.docs[:k]

    def similarity_search_with_relevance_scores(self, query, k):
        return [(d, self.score) for d in self.docs[:k]]


class _FakeReranker:
    def __init__(self, score_by_text):
        self.score_by_text = score_by_text

    def predict(self, pairs):
        return [self.score_by_text[text] for _, text in pairs]


def test_retrieve_node_falls_back_to_dense_only_without_reranker():
    """bm25/reranker를 안 넘기면(테스트·기존 호출부 호환) 예전 dense-only 경로를 그대로 탄다."""
    docs = [Document(page_content="본문", metadata={"source": "a.md"})]
    store = _FakeVectorstore(docs, score=0.77)
    retrieve_node, _, _, _ = make_nodes(store, None, None, None)

    result = retrieve_node({"question": "질문", "rewritten_question": ""})

    assert result["document"] == docs
    assert result["doc_scores"] == [0.77]


def test_retrieve_node_uses_rerank_when_bm25_and_reranker_given():
    dense_doc = Document(page_content="dense 결과", metadata={"source": "dense.md"})
    bm25_doc = Document(page_content="bm25 결과", metadata={"source": "bm25.md"})
    # BM25 코퍼스가 너무 작으면(N=2, df=1) rank_bm25 IDF 공식이 정확히 0이 되는 경계에
    # 걸린다(log((N-df+0.5)/(df+0.5))=log(1)=0) — 무관한 문서를 두 개 섞어서 피한다.
    distractors = [
        Document(page_content="전혀 다른 내용 하나", metadata={"source": "distractor1.md"}),
        Document(page_content="전혀 다른 내용 둘", metadata={"source": "distractor2.md"}),
    ]
    store = _FakeVectorstore([dense_doc])
    bm25_docs = [bm25_doc, *distractors]
    bm25 = build_bm25(bm25_docs)
    reranker = _FakeReranker({"dense 결과": 0.2, "bm25 결과": 0.95})

    retrieve_node, _, _, _ = make_nodes(
        store, None, None, None, bm25=bm25, bm25_docs=bm25_docs, reranker=reranker,
    )
    result = retrieve_node({"question": "bm25", "rewritten_question": ""})

    # 리랭커 점수가 더 높은 bm25 결과가 1등으로 올라와야 한다 — dense 원래 순위와 무관하게.
    assert result["document"][0].metadata["source"] == "bm25.md"
    assert result["doc_scores"][0] == 0.95


def test_grade_docs_drops_docs_below_threshold():
    docs = [_doc(0.9), _doc(0.6), _doc(0.1)]
    scores = [0.9, 0.6, 0.1]
    result = _grade_docs_node()({"document": docs, "doc_scores": scores})

    assert result["is_relevant"] is True
    assert result["document"] == [d for d, s in zip(docs, scores) if s > RELEVANCE_THRESHOLD]
    assert all(s > RELEVANCE_THRESHOLD for s in result["doc_scores"])


def test_grade_docs_keeps_all_when_all_above_threshold():
    docs = [_doc(0.9), _doc(0.8)]
    scores = [0.9, 0.8]
    result = _grade_docs_node()({"document": docs, "doc_scores": scores})

    assert result["document"] == docs
    assert result["doc_scores"] == scores


def test_grade_docs_top1_always_survives_when_relevant():
    """is_relevant=True는 top_score > threshold를 뜻하니, top-1은 필터링 후에도 항상 남아야 한다."""
    docs = [_doc(0.9), _doc(0.01)]
    scores = [0.9, 0.01]
    result = _grade_docs_node()({"document": docs, "doc_scores": scores})

    assert result["is_relevant"] is True
    assert docs[0] in result["document"]
    assert len(result["document"]) == 1
