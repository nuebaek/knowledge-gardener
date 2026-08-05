from app.core import catalog


def _write(tmp_path, rel, content="hello"):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_upsert_skips_unchanged_content(catalog_db):
    path = _write(catalog_db, "doc.md")
    assert catalog.upsert_document(path, source_type="writer", doc_type="til", title="t") is True
    # 내용을 안 바꾸고 다시 저장 → 아무 것도 갱신하지 않아야 "새/변경 파일만 처리"가 성립
    assert catalog.upsert_document(path, source_type="writer", doc_type="til", title="t") is False


def test_upsert_updates_on_change_and_resets_indexed_at(catalog_db):
    path = _write(catalog_db, "doc.md", "v1")
    catalog.upsert_document(path, source_type="writer", doc_type="til", title="t")

    with catalog._connect() as conn:
        conn.execute("UPDATE documents SET indexed_at = ? WHERE source_path = ?", ("2026-01-01", "doc.md"))

    path.write_text("v2", encoding="utf-8")
    assert catalog.upsert_document(path, source_type="writer", doc_type="til", title="t") is True

    row = catalog.get_document("doc.md")
    assert row["indexed_at"] is None  # 내용이 바뀌었으니 재인덱싱 대상으로 되돌아가야 함


def test_add_and_remove_tag_roundtrip(catalog_db):
    path = _write(catalog_db, "doc.md")
    catalog.upsert_document(path, source_type="writer", doc_type="til", title="t")

    assert catalog.add_tag("doc.md", "교육자료") == ["교육자료"]
    assert catalog.list_tags("doc.md") == ["교육자료"]
    assert catalog.remove_tag("doc.md", "교육자료") == []


def test_remove_tag_cleans_up_orphan(catalog_db):
    path = _write(catalog_db, "doc.md")
    catalog.upsert_document(path, source_type="writer", doc_type="til", title="t")
    catalog.add_tag("doc.md", "임시")
    catalog.remove_tag("doc.md", "임시")
    # 아무 문서도 안 쓰는 태그가 all_tags()에 죽은 채로 남으면 필터 UI가 지저분해짐
    assert catalog.all_tags() == []


def test_list_documents_filters_by_doc_type_and_tag(catalog_db):
    a = _write(catalog_db, "a.md", "a")
    b = _write(catalog_db, "b.md", "b")
    catalog.upsert_document(a, source_type="writer", doc_type="til", title="A")
    catalog.upsert_document(b, source_type="ingest", doc_type="processed", title="B")
    catalog.add_tag("a.md", "스크랩")

    assert {r["source_path"] for r in catalog.list_documents(doc_type="til")} == {"a.md"}
    assert {r["source_path"] for r in catalog.list_documents(tag="스크랩")} == {"a.md"}
    assert {r["source_path"] for r in catalog.list_documents()} == {"a.md", "b.md"}
