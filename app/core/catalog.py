import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import yaml

from app.core.paths import LIBRARY_DB, PROCESSED_DIR, PROJECT_ROOT, WRITER_DIR

BASE_DIR = PROJECT_ROOT
DB_PATH = LIBRARY_DB
_WRITER_TITLE_KEY = {"dailynote": "topic"}

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

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    source_path UNINDEXED, title, body, tokenize='trigram'
);
"""


@contextmanager
def _connect():
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


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    return parts[2].lstrip("\n") if len(parts) == 3 else text


def upsert_document(path: Path, *, source_type: str, doc_type: str, title: str) -> bool:
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
        conn.execute("DELETE FROM documents_fts WHERE source_path = ?", (source_path,))
        conn.execute(
            "INSERT INTO documents_fts (source_path, title, body) VALUES (?, ?, ?)",
            (source_path, title, strip_frontmatter(content)),
        )
    return True


def list_documents(*, doc_type: str | None = None, tag: str | None = None) -> list[sqlite3.Row]:
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


def prune_missing() -> list[str]:
    stale = [row["source_path"] for row in list_documents() if not (BASE_DIR / row["source_path"]).exists()]
    for source_path in stale:
        delete_document(source_path)
    return stale


def delete_document(source_path: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE source_path = ?", (source_path,))
        conn.execute("DELETE FROM documents_fts WHERE source_path = ?", (source_path,))


def fts_search(query: str, limit: int = 100) -> list[dict]:
    """trigram 매치를 관련도(rank)순으로, source_path/title을 반환한다. title까지 여기서
    바로 뽑는 이유: 매치 건마다 get_document()를 따로 부르면 _connect()가 매번 새 커넥션을
    여는 비용이 매치 수만큼 쌓여, 매치가 많은(흔한) 검색어에서는 그 비용이 이 함수로
    아끼려던 전수 스캔 비용을 넘어서 버린다(실측으로 확인). title은 이미 documents_fts
    컬럼에 있으니 조인 없이 한 번의 조회로 끝낸다."""
    literal = '"' + query.replace('"', '""') + '"'
    with _connect() as conn:
        rows = conn.execute(
            "SELECT source_path, title FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?",
            (literal, limit),
        ).fetchall()
    return [{"source_path": r["source_path"], "title": r["title"]} for r in rows]


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
        conn.execute(
            "DELETE FROM tags WHERE name = ? AND id NOT IN (SELECT tag_id FROM document_tags)",
            (name,),
        )
    return list_tags(source_path)


def _fallback_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped if len(stripped) <= 60 else f"{stripped[:60]}…"
    return path.stem


def _writer_title(path: Path, doc_type: str) -> str:
    try:
        raw = path.read_text(encoding="utf-8")
        _, frontmatter, _ = raw.split("---", 2)
        meta = yaml.safe_load(frontmatter)
        if not isinstance(meta, dict):
            raise ValueError("frontmatter가 dict가 아님")
    except (ValueError, yaml.YAMLError):
        return _fallback_title(path)

    if doc_type == "weeklynote":
        topics = meta.get("topics") or []
        return f"{meta.get('date', path.stem)} 주간노트" + (f" · {topics[0]}" if topics else "")
    if doc_type == "til":
        keywords = meta.get("keywords") or []
        if keywords:
            return keywords[0]
        what = str(meta.get("what", path.stem))
        return what if len(what) <= 60 else f"{what[:60]}…"
    key = _WRITER_TITLE_KEY.get(doc_type)
    value = str(meta.get(key, path.stem)) if key else path.stem
    return value if len(value) <= 60 else f"{value[:60]}…"


def _processed_title(stem: str) -> str:
    words = stem.replace("_", "-").split("-")
    return " ".join(w if w.isdigit() else w.capitalize() for w in words)


def sync_from_disk() -> dict:
    upserted = 0
    for doc_type_dir in sorted(p for p in WRITER_DIR.iterdir() if p.is_dir()) if WRITER_DIR.exists() else []:
        for path in sorted(doc_type_dir.glob("*.md")):
            upserted += upsert_document(
                path, source_type="writer", doc_type=doc_type_dir.name,
                title=_writer_title(path, doc_type_dir.name),
            )
    for path in sorted(PROCESSED_DIR.glob("**/*.md")) if PROCESSED_DIR.exists() else []:
        upserted += upsert_document(
            path, source_type="ingest", doc_type="processed", title=_processed_title(path.stem)
        )

    pruned = prune_missing()
    return {"upserted": upserted, "pruned": len(pruned)}
