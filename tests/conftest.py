import pytest

from app.core import catalog


@pytest.fixture
def catalog_db(tmp_path, monkeypatch):
    """카탈로그 테스트가 진짜 data/library.db를 안 건드리게 격리.

    BASE_DIR도 같이 옮기는 이유: catalog._rel()이 저장 경로를 BASE_DIR 기준 상대경로로
    바꾸는데, 테스트 파일이 tmp_path 밑에 있으니 BASE_DIR도 거기를 가리켜야 맞는다.
    """
    monkeypatch.setattr(catalog, "DB_PATH", tmp_path / "test_library.db")
    monkeypatch.setattr(catalog, "BASE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def isolated_writer(catalog_db, monkeypatch):
    """writer.save_docs()가 실제 data/writer/를 안 건드리고 catalog_db와 같은 tmp_path 밑에
    쓰게 만든다 — 두 BASE_DIR이 같은 트리 안에 있어야 catalog._rel()이 성립함."""
    from app.writer import writer

    writer_dir = catalog_db / "data" / "writer"
    monkeypatch.setattr(writer, "BASE_DIR", writer_dir)
    return writer_dir


@pytest.fixture
def isolated_corpus(catalog_db, monkeypatch):
    """app/services/corpus_service.py는 PROJECT_ROOT/ALLOWED_ROOTS를 자기 모듈 안에 따로
    갖고 있어서(catalog.py와는 별개 import) 이것도 같이 tmp_path로 옮겨야 API 테스트가
    실제 data/를 안 건드린다."""
    from app.services import corpus_service

    monkeypatch.setattr(corpus_service, "PROJECT_ROOT", catalog_db)
    monkeypatch.setattr(
        corpus_service,
        "ALLOWED_ROOTS",
        [(catalog_db / "data" / "processed").resolve(), (catalog_db / "data" / "writer").resolve()],
    )
    return catalog_db
