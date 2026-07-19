"""문서 카탈로그: writer/ingest가 만든 모든 마크다운 문서의 메타데이터 색인.

파일(.md)이 계속 원본이고, 이 SQLite는 그 파일들에 대한 색인일 뿐이다 — 그래야
DB를 지워도 backfill_catalog.py 한 번으로 복구되고, 사람이 파일을 직접 읽고 고치는
워크플로우가 안 깨진다. 그래서 여기엔 본문을 저장하지 않고 content_hash만 둔다.

upsert_document는 content_hash가 바뀐 파일만 갱신하고 그 자리에서 indexed_at을 NULL로
되돌린다. 이게 "새/변경 파일만 처리"의 실체다 — 매번 전체 디렉터리를 다시 훑는 대신,
그 파일을 실제로 저장한 시점(app/writer/writer.py, scripts/preprocess.py)에 1건만 반영한다.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from app.core.paths import LIBRARY_DB, PROJECT_ROOT

BASE_DIR = PROJECT_ROOT
DB_PATH = LIBRARY_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    source_path  TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    char_count   INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    indexed_at   TEXT
);

CREATE TABLE IF NOT EXISTS tags (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS document_tags (
    source_path TEXT NOT NULL REFERENCES documents(source_path) ON DELETE CASCADE,
    tag_id      INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (source_path, tag_id)
);

CREATE INDEX IF NOT EXISTS idx_documents_doc_type ON documents(doc_type);
"""


@contextmanager
def _connect():
    """호출부(writer/preprocess/corpus_service/백필 스크립트)가 제각각 진입점이라
    "누가 먼저 init을 불러줬는지"에 의존하면 순서 버그가 나기 쉽다. 그래서 스키마 보장을
    연결마다 이 안에서 해버린다 — CREATE TABLE IF NOT EXISTS라 비용은 무시할 만하다."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _rel(path: Path) -> str:
    return path.resolve().relative_to(BASE_DIR.resolve()).as_posix()


def upsert_document(path: Path, *, source_type: str, doc_type: str, title: str) -> bool:
    """path 내용으로 카탈로그를 갱신. 내용이 안 바뀌었으면 아무 것도 안 하고 False."""
    content = path.read_text(encoding="utf-8")
    content_hash = sha256(content.encode("utf-8")).hexdigest()
    source_path = _rel(path)
    now = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        row = conn.execute(
            "SELECT content_hash, created_at FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchone()
        if row and row["content_hash"] == content_hash:
            return False

        conn.execute(
            """
            INSERT INTO documents
                (source_path, title, source_type, doc_type, content_hash, char_count, created_at, updated_at, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(source_path) DO UPDATE SET
                title = excluded.title,
                content_hash = excluded.content_hash,
                char_count = excluded.char_count,
                updated_at = excluded.updated_at,
                indexed_at = NULL
            """,
            (
                source_path, title, source_type, doc_type, content_hash, len(content),
                row["created_at"] if row else now, now,
            ),
        )
    return True


def list_documents(*, doc_type: str | None = None, tag: str | None = None) -> list[sqlite3.Row]:
    """tags를 GROUP_CONCAT으로 같이 뽑아온다 — 목록 화면에서 문서당 태그 조회를 N+1로 안 만들려고."""
    query = """
        SELECT d.*, (
            SELECT GROUP_CONCAT(t.name) FROM document_tags dt JOIN tags t ON t.id = dt.tag_id
            WHERE dt.source_path = d.source_path
        ) AS tags
        FROM documents d
    """
    conditions, params = [], []
    if tag:
        conditions.append("""
            d.source_path IN (
                SELECT dt.source_path FROM document_tags dt JOIN tags t ON t.id = dt.tag_id
                WHERE t.name = ?
            )
        """)
        params.append(tag)
    if doc_type:
        conditions.append("d.doc_type = ?")
        params.append(doc_type)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY d.updated_at DESC"

    with _connect() as conn:
        return conn.execute(query, params).fetchall()


def pending_reindex() -> list[sqlite3.Row]:
    """indexed_at이 NULL인, 즉 저장/변경된 뒤로 아직 임베딩에 반영 안 된 문서들."""
    with _connect() as conn:
        return conn.execute("SELECT * FROM documents WHERE indexed_at IS NULL").fetchall()


def mark_indexed(source_path: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET indexed_at = ? WHERE source_path = ?",
            (datetime.now(timezone.utc).isoformat(), source_path),
        )


def get_document(source_path: str) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT * FROM documents WHERE source_path = ?", (source_path,)
        ).fetchone()


def delete_document(source_path: str) -> None:
    """파일을 직접 지운 뒤 카탈로그 row가 남아있으면 list_documents()가 그 파일을 읽으려다
    죽는다 — 파일을 지울 땐 항상 이것도 같이 불러야 한다. document_tags는 FK ON DELETE
    CASCADE라 같이 정리됨."""
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))


def list_tags(source_path: str) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT t.name FROM tags t JOIN document_tags dt ON dt.tag_id = t.id
            WHERE dt.source_path = ? ORDER BY t.name
            """,
            (source_path,),
        ).fetchall()
    return [r["name"] for r in rows]


def all_tags() -> list[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT name FROM tags ORDER BY name").fetchall()
    return [r["name"] for r in rows]


def add_tag(source_path: str, name: str) -> list[str]:
    with _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
        conn.execute(
            "INSERT OR IGNORE INTO document_tags (source_path, tag_id) "
            "SELECT ?, id FROM tags WHERE name = ?",
            (source_path, name),
        )
    return list_tags(source_path)


def remove_tag(source_path: str, name: str) -> list[str]:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM document_tags WHERE source_path = ? "
            "AND tag_id = (SELECT id FROM tags WHERE name = ?)",
            (source_path, name),
        )
        # 어느 문서도 안 쓰는 태그는 all_tags()/필터 목록에서 죽은 채로 남으니 같이 정리한다.
        conn.execute(
            "DELETE FROM tags WHERE name = ? AND id NOT IN (SELECT tag_id FROM document_tags)",
            (name,),
        )
    return list_tags(source_path)
