from datetime import date as _date, timedelta
import re

import yaml
from langchain_core.prompts import ChatPromptTemplate

from pathlib import Path
from rag import build_llm
from model import DailynoteEntry, WeeklynoteEntry, TilEntry

# ---------------- 프롬프트 ----------------

DAILY_NOTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a study-note writer. Turn a raw description of what someone learned in a lecture or course "
     "session into a structured daily learning note in Markdown. "
     "Match the language of the input (Korean input -> Korean output, English input -> English output).\n\n"
     "Output structure, in this exact order:\n"
     "1. A table of contents listing each distinct concept mentioned, numbered.\n"
     "2. One numbered section per concept, every section using exactly this structure:\n"
     "   - A one-line definition or summary of the concept.\n"
     "   - Why or when this concept is used.\n"
     "   - Key details wrapped in a collapsible toggle using this exact HTML: "
     "`<details><summary>핵심</summary>` on one line, the content, then `</details>` "
     "(keep the summary label in the input's language).\n"
     "   - If there is something worth exploring further, add a short \"더 알아보면 좋을 내용\" note "
     "inside the same section. Omit this part entirely if there is nothing to add — do not force it.\n"
     "3. A final section with NO bullet points: 2-4 flowing prose paragraphs that weave every concept "
     "learned that day into one continuous narrative.\n\n"
     "Use only what is explicitly stated in the input. Do not invent concepts, definitions, or facts."),
    ("human",
     "Topic: {topic}\n"
     "Concepts mentioned: {related_concepts}\n"
     "Raw description of what was learned:\n{learned}"),
])


WEEKLY_NOTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a study-note writer. Synthesize a week's worth of daily learning notes into one concise "
     "weekly summary in Markdown. Match the language of the input notes. "
     "Be noticeably more concise than the daily notes — someone should be able to read only this document "
     "and immediately understand what was learned that week, without opening the daily notes.\n\n"
     "Output structure:\n"
     "1. A synthesized overview of the week's concepts and topics, organized by theme rather than by day — "
     "merge related concepts that appear across different days into one entry instead of repeating them.\n"
     "2. A final section with NO bullet points: 2-3 flowing prose paragraphs reading as one continuous "
     "narrative of the week's learning, in the same style as a daily note's closing summary.\n\n"
     "Use only what is explicitly stated in the daily notes provided. Do not invent content."),
    ("human",
     "Week: {week_start} ~ {week_end}\n\n"
     "This week's daily notes:\n{daily_notes}"),
])


TIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a retrospective-note writer. Turn structured retrospective input into a clean, generic "
     "TIL-style (Today I Learned) retrospective document in Markdown. This must work equally well for a "
     "personal project, a team project, a single day's retrospective, or a dev journal entry — do not "
     "assume any specific category. Match the language of the input.\n\n"
     "Output structure:\n"
     "1. **What happened** — a brief account of the situation or work, from the input.\n"
     "2. **What I learned** — the core takeaway.\n"
     "3. **What I got stuck on** — the problem and how it was resolved. Omit this section if that input is empty.\n"
     "4. **Reflection** — the reflection input, expanded into full sentences if it was terse.\n"
     "5. **Next action** — the action-plan input. Omit this section if that input is empty.\n"
     "6. A short keyword list at the end.\n\n"
     "Keep the core of each section solid enough that reading it back later immediately tells the reader "
     "what was learned and reflected on, regardless of category. Use only what is explicitly given."),
    ("human",
     "What happened: {what}\n"
     "Learned: {learned}\n"
     "Troubleshooting: {troubleshooting}\n"
     "Reflection: {reflection}\n"
     "Next action: {actionplan}\n"
     "Keywords: {keywords}"),
])


# ---------------- writer 정의 ----------------
llm = build_llm()

BASE_DIR = Path(__file__).parent / "data" / "writer"
today = _date.today()

def slugify(text: str, max_len: int = 20) -> str:
    slug = re.sub(r"\s+", "-", text.strip())
    slug = re.sub(r"[^\w\-가-힣]", "", slug)
    return slug[:max_len].strip("-") or "untitled"


def _read_frontmatter(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    _, frontmatter, _ = raw.split("---", 2)
    return yaml.safe_load(frontmatter)


def save_docs(dir, filename, entry, body) -> Path:
    frontmatter = yaml.safe_dump(entry.model_dump(mode="json"), allow_unicode=True)
    content = f"---\n{frontmatter}---\n\n{body}\n"

    out_path = BASE_DIR / dir / f"{filename}.md"
    counter = 2
    while out_path.exists():
        out_path = BASE_DIR / dir / f"{filename}-{counter}.md"
        counter += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


# daily note
def write_daily_note(
        topic: str,
        learned: str,
        related_concepts: list[str] | None
) -> str:
    entry = DailynoteEntry(
        date=today,
        topic=topic,
        learned=learned,
        related_concepts=related_concepts or [],
    )

    messages = DAILY_NOTE_PROMPT.invoke({
        "topic": topic,
        "related_concepts": ", ".join(related_concepts) if related_concepts else "(없음)",
        "learned": learned,
    }).to_messages()

    response = llm.invoke(messages)

    slug = slugify(topic)
    out_path = save_docs("dailynote", f"{today}-{slug}", entry, response.content)
    return f"저장 완료: {out_path}"


# weekly note
def write_weekly_note(as_of: str | None = None) -> str:
    ref_date = _date.fromisoformat(as_of) if as_of else today
    monday = ref_date - timedelta(days=ref_date.weekday())
    sunday = monday + timedelta(days=6)

    daily_dir = BASE_DIR / "dailynote"
    week_files = []
    for p in sorted(daily_dir.glob("*.md")):
        try:
            file_date = _date.fromisoformat(p.stem[:10])
        except ValueError:
            continue
        if monday <= file_date <= sunday:
            week_files.append(p)

    if not week_files:
        return f"{monday}~{sunday} 사이 저장된 daily note가 없음"

    topics, related_concepts, daily_bodies = [], [], []
    for p in week_files:
        meta = _read_frontmatter(p)
        if meta.get("topic") and meta["topic"] not in topics:
            topics.append(meta["topic"])
        for c in meta.get("related_concepts", []):
            if c not in related_concepts:
                related_concepts.append(c)
        daily_bodies.append(p.read_text(encoding="utf-8"))

    entry = WeeklynoteEntry(
        date=monday,
        topics=topics,
        related_concepts=related_concepts,
    )

    messages = WEEKLY_NOTE_PROMPT.invoke({
        "week_start": monday.isoformat(),
        "week_end": sunday.isoformat(),
        "daily_notes": "\n\n---\n\n".join(daily_bodies),
    }).to_messages()

    response = llm.invoke(messages)

    out_path = save_docs("weeklynote", monday.isoformat(), entry, response.content)
    return f"저장 완료: {out_path}"


# til
def write_tilnote(
    what: str,
    learned: str,
    troubleshooting: str,
    reflection: str,
    actionplan: str,
    keywords: list[str],
) -> str:
    entry = TilEntry(
        date=today,
        what=what,
        learned=learned,
        troubleshooting=troubleshooting,
        reflection=reflection,
        actionplan=actionplan,
        keywords=keywords,
    )

    messages = TIL_PROMPT.invoke({
        "what": what,
        "learned": learned,
        "troubleshooting": troubleshooting,
        "reflection": reflection,
        "actionplan": actionplan,
        "keywords": keywords,
    }).to_messages()

    response = llm.invoke(messages)

    slug = slugify(what)
    out_path = save_docs("til", f"{today}-{slug}", entry, response.content)
    return f"저장 완료: {out_path}"
