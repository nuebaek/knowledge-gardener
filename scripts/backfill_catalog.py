"""data/writer, data/processed를 카탈로그와 수동으로 맞추고 싶을 때 쓰는 CLI.

앱을 재시작하면 app/main.py의 lifespan이 catalog.sync_from_disk()를 자동으로 호출해서
이 스크립트 없이도 새/삭제된 파일이 반영된다 — 이건 "서버 재시작 없이 지금 바로" 맞추고
싶을 때만 수동으로 돌리는 용도.

실행:
    uv run python scripts/backfill_catalog.py (kaia-project 루트에서)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # scripts/는 app 패키지 바깥이라 루트를 직접 넣어줘야 import된다

from app.core import catalog


def main() -> None:
    result = catalog.sync_from_disk()
    print(f"동기화 완료: upsert {result['upserted']}개, 삭제(고아 row 정리) {result['pruned']}개 → {catalog.DB_PATH}")


if __name__ == "__main__":
    main()
