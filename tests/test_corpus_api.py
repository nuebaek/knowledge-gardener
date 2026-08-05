import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.core import catalog
from app.services import corpus_service as corpus_controller
from app.api.routes.corpus import router


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _seed(base, rel, content, doc_type, source_type, title):
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    catalog.upsert_document(path, source_type=source_type, doc_type=doc_type, title=title)
    return rel


def test_documents_list_includes_writer_and_ingest_docs(isolated_corpus):
    _seed(isolated_corpus, "data/writer/til/a.md", "til content", "til", "writer", "A")
    _seed(isolated_corpus, "data/processed/b.md", "processed content", "processed", "ingest", "B")

    res = _client().get("/documents")
    assert res.status_code == 200
    assert {d["id"] for d in res.json()} == {"data/writer/til/a.md", "data/processed/b.md"}


def test_documents_filter_by_doc_type(isolated_corpus):
    _seed(isolated_corpus, "data/writer/til/a.md", "til content", "til", "writer", "A")
    _seed(isolated_corpus, "data/processed/b.md", "processed content", "processed", "ingest", "B")

    res = _client().get("/documents", params={"doc_type": "til"})
    assert [d["id"] for d in res.json()] == ["data/writer/til/a.md"]


def test_tag_add_filter_remove_roundtrip(isolated_corpus):
    doc_id = _seed(isolated_corpus, "data/writer/til/a.md", "til content", "til", "writer", "A")
    client = _client()

    detail = client.get(f"/documents/{doc_id}").json()
    assert detail["tags"] == []

    assert client.post(f"/documents/{doc_id}/tags", json={"name": "교육자료"}).json() == ["교육자료"]
    assert [d["id"] for d in client.get("/documents", params={"tag": "교육자료"}).json()] == [doc_id]

    assert client.delete(f"/documents/{doc_id}/tags/교육자료").json() == []
    assert client.get("/tags").json() == []  # orphan 태그도 같이 정리돼야 함


def test_get_document_not_in_catalog_returns_404(isolated_corpus):
    res = _client().get("/documents/does/not/exist.md")
    assert res.status_code == 404


def test_resolve_rejects_paths_outside_allowed_roots(isolated_corpus):
    """URL 레이어(TestClient)는 "../" 를 브라우저/클라이언트 단에서 정규화해버려서 진짜
    경로 탈출 시도를 흉내내기 어렵다 — 그래서 가드 함수(_resolve)를 직접 호출해서 검증."""
    with pytest.raises(HTTPException) as exc:
        corpus_controller._resolve("../../../../etc/passwd")
    assert exc.value.status_code == 404


def test_ask_about_document_returns_answer(isolated_corpus, monkeypatch):
    _seed(
        isolated_corpus, "data/processed/a.md",
        "---\ntitle: A\n---\ncontent about topic X",
        "processed", "ingest", "A",
    )
    monkeypatch.setattr(
        corpus_controller,
        "answer_document_question",
        lambda question, document_text, history: f"answered: {question} | doc={document_text} | history={history}",
    )

    res = _client().post(
        "/documents/data/processed/a.md/ask",
        json={"question": "X가 뭐야?", "history": [{"role": "user", "content": "이전 질문"}]},
    )

    assert res.status_code == 200
    body = res.json()
    assert "X가 뭐야?" in body["answer"]
    assert "content about topic X" in body["answer"]
    assert "이전 질문" in body["answer"]
    assert "title: A" not in body["answer"]  # frontmatter는 get_document()가 이미 벗겨서 넘겨야 함


def test_ask_about_document_404_for_missing_doc(isolated_corpus):
    res = _client().post(
        "/documents/data/processed/missing.md/ask",
        json={"question": "hi", "history": []},
    )
    assert res.status_code == 404


def test_ask_about_document_defaults_history_to_empty(isolated_corpus, monkeypatch):
    _seed(isolated_corpus, "data/processed/a.md", "content", "processed", "ingest", "A")
    monkeypatch.setattr(
        corpus_controller, "answer_document_question", lambda question, document_text, history: "ok"
    )

    res = _client().post("/documents/data/processed/a.md/ask", json={"question": "hi"})

    assert res.status_code == 200
    assert res.json() == {"answer": "ok"}
