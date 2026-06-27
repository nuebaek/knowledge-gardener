from pathlib import Path

from markitdown import MarkItDown

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUT_DIR = Path(__file__).parent / "data" / "processed"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_paths:
        print(f"data/raw 에 PDF가 없습니다: {RAW_DIR}")
        return

    md = MarkItDown()
    print(f"변환 대상 PDF: {len(pdf_paths)}개")
    for pdf in pdf_paths:
        result = md.convert(str(pdf))
        out_path = OUT_DIR / f"{pdf.stem}.md"
        out_path.write_text(result.text_content, encoding="utf-8")
        print(f"  {pdf.name} -> {out_path.name}")

    print(f"전처리 완료 → {OUT_DIR}")


if __name__ == "__main__":
    main()
