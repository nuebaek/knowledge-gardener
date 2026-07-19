import re
from pathlib import Path

from fastapi import HTTPException

from app.core import catalog
from app.core.paths import PROCESSED_DIR, PROJECT_ROOT, WRITER_DIR
from app.schemas.corpus import CorpusResponse, DocumentDetail, DocumentSummary, SearchHit, SearchResponse

# 카탈로그에 없는 임의 경로(예: ../../.env)가 넘어와도 이 두 트리 밖은 절대 못 읽도록 고정.
ALLOWED_ROOTS = [PROCESSED_DIR.resolve(), WRITER_DIR.resolve()]

_MARKUP = re.compile(r"[*_`#]")
_HEADING = re.compile(r"^#{1,6}\s+(.*)")
_TOC_LABEL = re.compile(r"^(table of contents|목차)\s*:?\s*$", re.IGNORECASE)
_LIST_ITEM = re.compile(r"^[*+-]\s")
_LINK_ONLY = re.compile(r"^\[[^\]]+\]\([^)]*\)\s*$")


def _excerpt(text: str, limit: int = 140) -> str:
    """헤더/TOC 라벨/링크/불릿 줄을 모두 건너뛰고 진짜 본문 문단만 뽑아 목록 카드에 쓴다."""
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
    path = (PROJECT_ROOT / doc_id).resolve()
    if not any(root == path or root in path.parents for root in ALLOWED_ROOTS) or not path.exists():
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return path


def _to_summary(row) -> DocumentSummary:
    text = (PROJECT_ROOT / row["source_path"]).read_text(encoding="utf-8")
    return DocumentSummary(
        id=row["source_path"],
        title=row["title"],
        excerpt=_excerpt(text),
        char_count=row["char_count"],
        doc_type=row["doc_type"],
        tags=row["tags"].split(",") if row["tags"] else [],
    )


def get_corpus() -> CorpusResponse:
    rows = catalog.list_documents(doc_type="processed")
    return CorpusResponse(count=len(rows), topics=[r["title"] for r in rows])


def list_documents(doc_type: str | None = None, tag: str | None = None) -> list[DocumentSummary]:
    return [_to_summary(row) for row in catalog.list_documents(doc_type=doc_type, tag=tag)]


def get_document(doc_id: str) -> DocumentDetail:
    row = catalog.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    text = _resolve(doc_id).read_text(encoding="utf-8")
    return DocumentDetail(
        id=doc_id,
        title=row["title"],
        content=text,
        char_count=row["char_count"],
        doc_type=row["doc_type"],
        tags=catalog.list_tags(doc_id),
    )


def add_tag(doc_id: str, name: str) -> list[str]:
    if catalog.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return catalog.add_tag(doc_id, name.strip())


def remove_tag(doc_id: str, name: str) -> list[str]:
    if catalog.get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    return catalog.remove_tag(doc_id, name)


def list_tags() -> list[str]:
    return catalog.all_tags()


def search_documents(query: str, limit: int = 20) -> SearchResponse:
    """문서 전문에서 문단 단위 키워드 검색. 가장 가까운 헤딩을 section으로 붙인다."""
    q = query.strip()
    if not q:
        return SearchResponse(query=query, hits=[])
    q_lower = q.lower()

    hits: list[SearchHit] = []
    for row in catalog.list_documents():
        if len(hits) >= limit:
            break
        text = (PROJECT_ROOT / row["source_path"]).read_text(encoding="utf-8")
        title = row["title"]
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
                    doc_id=row["source_path"],
                    title=title,
                    section=current_section,
                    snippet=f"{prefix}{snippet}{'…' if window_end < len(clean) else ''}",
                    match_start=match_start,
                    match_end=match_end,
                )
            )

    return SearchResponse(query=q, hits=hits)
