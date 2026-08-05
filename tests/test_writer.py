from datetime import date

from app.writer import model, writer


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    """실제 API 호출 없이 write_weekly_note()의 llm.invoke(...)만 대체 — 이 테스트가
    확인하려는 건 파일 저장 방식(중복 여부)이라 생성 내용 품질은 관심사가 아님."""

    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        return _FakeResponse(self.content)


def test_save_docs_increments_on_collision_by_default(isolated_writer):
    entry = model.TilEntry(
        date=date.today(), what="a", learned="", troubleshooting="", reflection="", actionplan="", keywords=[]
    )
    p1 = writer.save_docs("til", "2026-07-19-x", entry, "body1", title="t")
    p2 = writer.save_docs("til", "2026-07-19-x", entry, "body2", title="t")
    assert p1 != p2
    assert p1.exists() and p2.exists()


def test_save_docs_overwrite_replaces_same_file(isolated_writer):
    entry = model.WeeklynoteEntry(date=date.today(), topics=[], related_concepts=[])
    p1 = writer.save_docs("weeklynote", "2026-07-13", entry, "body1", title="t", overwrite=True)
    p2 = writer.save_docs("weeklynote", "2026-07-13", entry, "body2", title="t", overwrite=True)

    assert p1 == p2
    assert "body2" in p2.read_text(encoding="utf-8")
    assert len(list((isolated_writer / "weeklynote").glob("*.md"))) == 1


def test_write_weekly_note_regenerate_does_not_duplicate(isolated_writer, monkeypatch):
    """이번에 고친 버그의 회귀 테스트: 같은 주를 두 번 생성해도 파일/카탈로그 row가 하나여야 함."""
    monday = date.fromisoformat("2026-07-13")
    daily_entry = model.DailynoteEntry(date=monday, topic="LoRA", learned="...", related_concepts=["PEFT"])
    writer.save_docs("dailynote", "2026-07-13-lora", daily_entry, "daily body", title="LoRA")

    monkeypatch.setattr(writer, "default_llm", lambda: _FakeLLM("weekly body v1"))
    writer.write_weekly_note(as_of="2026-07-13")

    monkeypatch.setattr(writer, "default_llm", lambda: _FakeLLM("weekly body v2"))
    writer.write_weekly_note(as_of="2026-07-13")

    weekly_files = list((isolated_writer / "weeklynote").glob("*.md"))
    assert len(weekly_files) == 1
    assert "weekly body v2" in weekly_files[0].read_text(encoding="utf-8")

    from app.core import catalog

    rows = catalog.list_documents(doc_type="weeklynote")
    assert len(rows) == 1
