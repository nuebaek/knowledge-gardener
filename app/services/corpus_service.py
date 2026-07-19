import re
from pathlib import Path

from fastapi import HTTPException

from app.core.paths import PROCESSED_DIR as DATA_DIR
from app.schemas.corpus import CorpusResponse, DocumentDetail, DocumentSummary, SearchHit, SearchResponse

_MARKUP = re.compile(r"[*_`#]")
_HEADING = re.compile(r"^#{1,6}\s+(.*)")
_TOC_LABEL = re.compile(r"^(table of contents|목차)\s*:?\s*$", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^[*+-]\s")
_LINK_ONLY = re.compile(r"^\[[^\]]+\]\([^)]*\)\s*$")


def _humanize(stem: str) -> str:
    words = stem.replace("_", "-").split("-")
    return " ".join(w if w.isdigit() else w.capitalize() for w in words)


def _paths() -> list[Path]:
    return sorted(DATA_DIR.glob("*.md"))


def _excerpt(text: str, limit: int = 140) -> str:
    """헤더/TOC 라벨/링크/불릿 줄을 모두 건너뛰고 진짜 본문 문단만 뽑아 목록 카드에 쓴다.

    CS231n 원문 대부분이 헤더 바로 다음 줄에 "Table of Contents:" + 중첩 링크 불릿을
    두고 있어, 이 블록 전체를 지나쳐야 실제 소개 문단에 닿는다.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("!["):
            continue
        if _TOC_LABEL.match(stripped) or _LIST_ITEM.match(stripped) or _LINK_ONLY.match(stripped):
            continue
        clean = _MARKUP.sub("", stripped).strip()
        if not clean:
            continue
        return clean[:limit] + ("…" if len(clean) > limit else "")
    return ""


def _resolve(doc_id: str) -> Path:
    path = (DATA_DIR / f"{doc_id}.md").resolve()
    if DATA_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return path


def get_corpus() -> CorpusResponse:
    paths = _paths()
    topics = [_humanize(p.stem) for p in paths]
    return CorpusResponse(count=len(paths), topics=topics)


def list_documents() -> list[DocumentSummary]:
    summaries = []
    for path in _paths():
        text = path.read_text(encoding="utf-8")
        summaries.append(
            DocumentSummary(
                id=path.stem,
                title=_humanize(path.stem),
                excerpt=_excerpt(text),
                char_count=len(text),
            )
        )
    return summaries


def get_document(doc_id: str) -> DocumentDetail:
    path = _resolve(doc_id)
    text = path.read_text(encoding="utf-8")
    return DocumentDetail(id=doc_id, title=_humanize(doc_id), content=text, char_count=len(text))


def search_documents(query: str, limit: int = 20) -> SearchResponse:
    """문서 전문에서 문단 단위 키워드 검색. 가장 가까운 헤딩을 section으로 붙인다."""
    q = query.strip()
    if not q:
        return SearchResponse(query=query, hits=[])
    q_lower = q.lower()

    hits: list[SearchHit] = []
    for path in _paths():
        if len(hits) >= limit:
            break
        text = path.read_text(encoding="utf-8")
        title = _humanize(path.stem)
        current_section = title

        for block in text.split("\n\n"):
            if len(hits) >= limit:
                break
            block = block.strip()
            if not block:
                continue

            heading_match = _HEADING.match(block)
            if heading_match:
                current_section = heading_match.group(1).strip()
                continue

            clean = _MARKUP.sub("", block).strip()
            match_pos = clean.lower().find(q_lower)
            if match_pos == -1:
                continue

            window_start = max(0, match_pos - 60)
            window_end = min(len(clean), match_pos + len(q) + 100)
            snippet = clean[window_start:window_end]

            prefix = "…" if window_start > 0 else ""
            match_start = match_pos - window_start + len(prefix)
            match_end = match_start + len(q)

            hits.append(
                SearchHit(
                    doc_id=path.stem,
                    title=title,
                    section=current_section,
                    snippet=f"{prefix}{snippet}{'…' if window_end < len(clean) else ''}",
                    match_start=match_start,
                    match_end=match_end,
                )
            )

    return SearchResponse(query=q, hits=hits)
