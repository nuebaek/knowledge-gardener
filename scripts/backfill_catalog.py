"""catalog.py 도입 전에 이미 쌓여있던 data/processed, data/writer 파일을 1회성으로 카탈로그에 채운다.

이후로는 app/writer/writer.py, scripts/preprocess.py가 저장 시점에 알아서 upsert하므로
이 스크립트는 매번 돌릴 필요가 없다 — DB 파일을 지우고 새로 만들 때만 다시 실행하면 됨.

실행:
    uv run python scripts/backfill_catalog.py (kaia-project 루트에서)
"""
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # scripts/는 app 패키지 바깥이라 루트를 직접 넣어줘야 import된다

from app.core import catalog
from app.core.paths import PROCESSED_DIR, WRITER_DIR

DOC_TYPE_TITLE_KEY = {"dailynote": "topic", "til": "what"}  # weeklynote는 별도 처리


def _writer_title(path: Path, doc_type: str) -> str:
    raw = path.read_text(encoding="utf-8")
    _, frontmatter, _ = raw.split("---", 2)
    meta = yaml.safe_load(frontmatter) or {}
    if doc_type == "weeklynote":
        topics = meta.get("topics") or []
        return f"{meta.get('date', path.stem)} 주간노트" + (f" · {topics[0]}" if topics else "")
    key = DOC_TYPE_TITLE_KEY.get(doc_type)
    value = str(meta.get(key, path.stem)) if key else path.stem
    return value if len(value) <= 60 else f"{value[:60]}…"


def _processed_title(stem: str) -> str:
    words = stem.replace("_", "-").split("-")
    return " ".join(w if w.isdigit() else w.capitalize() for w in words)


def main() -> None:
    added = skipped = 0

    for doc_type_dir in sorted(p for p in WRITER_DIR.iterdir() if p.is_dir()) if WRITER_DIR.exists() else []:
        for path in sorted(doc_type_dir.glob("*.md")):
            changed = catalog.upsert_document(
                path, source_type="writer", doc_type=doc_type_dir.name,
                title=_writer_title(path, doc_type_dir.name),
            )
            added += changed
            skipped += not changed

    for path in sorted(PROCESSED_DIR.glob("**/*.md")) if PROCESSED_DIR.exists() else []:
        changed = catalog.upsert_document(
            path, source_type="ingest", doc_type="processed", title=_processed_title(path.stem)
        )
        added += changed
        skipped += not changed

    print(f"백필 완료: 신규/갱신 {added}개, 변경 없음(스킵) {skipped}개 → {catalog.DB_PATH}")


if __name__ == "__main__":
    main()
