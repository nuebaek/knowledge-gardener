from pathlib import Path

from schemas import CorpusResponse

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _humanize(stem: str) -> str:
    words = stem.replace("_", "-").split("-")
    return " ".join(w if w.isdigit() else w.capitalize() for w in words)


def get_corpus() -> CorpusResponse:
    paths = sorted(DATA_DIR.glob("*.md"))
    topics = [_humanize(p.stem) for p in paths]
    return CorpusResponse(count=len(paths), topics=topics)
