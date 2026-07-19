"""data/raw/** 의 모든 파일을 markitdown으로 변환해 data/processed/**/*.md 로 저장.

지원 형식: markitdown이 처리 가능한 모든 형식 (pdf, html, docx, pptx, xlsx, csv, …).
변환에 실패한 파일은 건너뛰고 이유를 출력한다.
디렉터리 구조는 raw → processed 로 그대로 유지한다.

실행:
    uv run python preprocess.py
"""
from pathlib import Path

from markitdown import MarkItDown

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUT_DIR = Path(__file__).parent / "data" / "processed"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in sorted(RAW_DIR.rglob("*"))
                  if p.is_file() and not p.name.startswith(".")]
    if not candidates:
        print(f"data/raw 에 파일이 없습니다: {RAW_DIR}")
        return

    md = MarkItDown()
    print(f"변환 대상: {len(candidates)}개")
    ok, skip = 0, 0
    for src in candidates:
        rel = src.relative_to(RAW_DIR)          # raw 기준 상대 경로
        out = (OUT_DIR / rel).with_suffix(".md") # processed 에 동일 구조로
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = md.convert(str(src))
            out.write_text(result.text_content, encoding="utf-8")
            print(f"  ✓ {rel} → {out.relative_to(OUT_DIR)}")
            ok += 1
        except Exception as e:
            print(f"  ✗ {rel} 변환 실패: {e}")
            skip += 1

    print(f"\n완료: 성공 {ok}개 / 실패(건너뜀) {skip}개 → {OUT_DIR}")


if __name__ == "__main__":
    main()
