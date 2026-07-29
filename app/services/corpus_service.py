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
_LIST_ITEM = re.compile(r"^(?:[*+-]|\d+(?:\.\d+)*\.?)\s")
_LINK_ONLY = re.compile(r"^\[[^\]]+\]\([^)]*\)\s*$")
_HR = re.compile(r"^(-{3,}|\*{3,}|_{3,})$")


def _strip_frontmatter(text: str) -> str:
    """writer가 저장하는 문서는 전부 YAML frontmatter(`---`로 감싼 블록)로 시작한다 —
    title/tags/doc_type은 이미 별도 필드로 카탈로그에 있으니 사용자 화면(본문/미리보기)엔
    이 블록을 보여줄 이유가 없다. `---`로 시작하지 않는 문서(ingest 변환본 등)는 그대로 둔다."""
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    return parts[2].lstrip("\n") if len(parts) == 3 else text


def _excerpt(text: str, limit: int = 140) -> str:
    """헤더/TOC 라벨/링크/불릿 줄을 모두 건너뛰고 진짜 본문 문단만 뽑아 목록 카드에 쓴다."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("!["):
            continue
        if _TOC_LABEL.match(stripped) or _LIST_ITEM.match(stripped) or _LINK_ONLY.match(stripped) or _HR.match(stripped):
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


def _to_summary(row) -> DocumentSummary | None:
    path = PROJECT_ROOT / row["source_path"]
    if not path.exists():
        return None  # 카탈로그엔 있는데 파일이 직접 지워진 경우 — 목록에서 조용히 빠짐
    text = _strip_frontmatter(path.read_text(encoding="utf-8"))
    return DocumentSummary(
        id=row["source_path"],
        title=row["title"],
        excerpt=_excerpt(text),
        char_count=row["char_count"],
        created_at=row["created_at"],
        doc_type=row["doc_type"],
        tags=row["tags"].split(",") if row["tags"] else [],
    )


def get_corpus() -> CorpusResponse:
    rows = catalog.list_documents(doc_type="processed")
    return CorpusResponse(count=len(rows), topics=[r["title"] for r in rows])


def list_documents(doc_type: str | None = None, tag: str | None = None) -> list[DocumentSummary]:
    summaries = (_to_summary(row) for row in catalog.list_documents(doc_type=doc_type, tag=tag))
    return [s for s in summaries if s is not None]


def get_document(doc_id: str) -> DocumentDetail:
    row = catalog.get_document(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail="문서를 찾을 수 없습니다.")
    text = _strip_frontmatter(_resolve(doc_id).read_text(encoding="utf-8"))
    return DocumentDetail(
        id=doc_id,
        title=row["title"],
        content=text,
        char_count=row["char_count"],
        created_at=row["created_at"],
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
        path = PROJECT_ROOT / row["source_path"]
        if not path.exists():
            continue  # 카탈로그엔 있는데 파일이 직접 지워진 경우
        text = _strip_frontmatter(path.read_text(encoding="utf-8"))
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
            break  # 문서당 첫 매치 하나만 — 안 그러면 문단마다 같은 문서가 중복으로 뜬다

    return SearchResponse(query=q, hits=hits)
