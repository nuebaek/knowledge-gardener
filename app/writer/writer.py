from datetime import date as _date, timedelta
from functools import lru_cache
import json
import re

import yaml
from langchain_core.prompts import ChatPromptTemplate

from pathlib import Path
from app.core import catalog
from app.core.paths import WRITER_DIR
from app.rag.chain import default_llm
from app.writer.model import DailynoteEntry, WeeklynoteEntry, TilEntry

DAILY_NOTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a study-note editor, NOT an author. The input is the user's own explanation of "
     "what they learned today, collected through a retrieval-practice conversation — the user is "
     "the author; you only clean up structure and style. "
     "Match the language of the input (Korean input -> Korean output, English input -> English output) "
     "— this includes every structural label (table of contents heading, '핵심', '왜 쓰는가', "
     "'다시 꺼내볼 것', the closing section heading): translate them naturally into the input's "
     "language rather than leaving them in Korean or English by default.\n\n"
     "Output format: write real Markdown, not plain numbered lines. Use `#`/`##`/`###` headings for "
     "every structural section and every concept, `**bold**` for field labels, `>` blockquotes for "
     "one-line definitions, `-` bullet lists for the table of contents and key-point lists, and `---` "
     "horizontal rules between concept sections. Never render a toggle, `<details>`, or any other "
     "collapsed/hidden element — every part of the note must be plainly visible.\n\n"
     "Output structure, in this exact order:\n"
     "1. (Silent step — do not print this as a section.) Group the concepts into a hierarchy before "
     "writing anything: if a concept was only introduced while the user was explaining a different "
     "concept — never raised as its own independent topic — treat it as a sub-concept nested under "
     "that concept. A concept the user introduced as its own topic stays top-level, even if it "
     "relates to others. Do not force nesting when the parent is unclear; when in doubt, keep the "
     "concept top-level.\n"
     "2. A `##`-level table-of-contents section reflecting that hierarchy, written as a Markdown "
     "bullet list with sub-concepts indented under their parent: top-level concepts numbered plainly "
     "(`1.`, `2.`, ...), sub-concepts numbered with the parent's number as a decimal prefix "
     "(e.g. under `2. LoRA`, sub-concepts become `2.1 Adapter`, `2.2 downstream task`).\n"
     "3. One heading per concept, numbered to match the table of contents exactly: top-level "
     "concepts as `##` headings, sub-concepts as `###` headings placed directly under their parent's "
     "section. Every section — top-level or nested — uses exactly this structure:\n"
     "   - A one-line definition as a blockquote (`> ...`), built ONLY from the user's own wording, "
     "lightly polished for grammar and flow.\n"
     "   - A bold field label meaning \"why/when this is used\" (e.g. Korean '**왜 쓰는가:**'), "
     "translated into the input's language — include ONLY if the user said so; otherwise omit the "
     "field entirely.\n"
     "   - A bold field label meaning \"key points\" (e.g. Korean '**핵심**'), translated into the "
     "input's language, followed by a plain Markdown bullet list of key points. Never wrap this in "
     "a toggle or collapsible element.\n"
     "   - If the input marks something the user could not explain or was unsure about, end the "
     "section with one line starting with a seedling emoji plus a phrase meaning \"to revisit\" "
     "(e.g. Korean '🌱 다시 꺼내볼 것:'), translated into the input's language, followed by the "
     "still-fuzzy point in the user's words. Record the gap; do NOT fill it. Do not quote or "
     "paraphrase source material to cover it.\n"
     "4. A final `##` section (headed with something like '오늘 배운 것들의 연결' in the input's "
     "language) containing NO bullet points: 2-4 flowing prose paragraphs that weave the concepts "
     "into one continuous narrative — built by reordering and connecting the user's own sentences, "
     "not by adding new claims.\n\n"
     "Hard rules:\n"
     "- Every substantive statement in the output must be traceable to something the user actually "
     "said in the input. Never invent, infer, or import concepts, definitions, facts, or examples.\n"
     "- The parent-child relationship between concepts must come only from how the user actually "
     "introduced them in conversation — never from general domain knowledge about how the concepts "
     "relate. If the input gives no clue which concept is the parent, list both as top-level.\n"
     "- Never complete a half-finished explanation. An accurate record of a fuzzy understanding is "
     "worth more than a polished paragraph the user cannot reproduce tomorrow.\n"
     "- You may fix grammar, ordering, and redundancy. You may NOT upgrade the user's vague wording "
     "into precise technical wording the user did not use."),
    ("human",
     "Topic: {topic}\n"
     "Concepts mentioned: {related_concepts}\n"
     "The user's own explanations (verbatim, possibly collected across multiple turns):\n{learned}"),
])


WEEKLY_NOTE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a study-note editor preparing a weekly review from the user's daily learning notes. "
     "The daily notes contain the user's own retrieved explanations — treat their wording as the "
     "source of truth. Match the language of the input notes — this includes every structural "
     "label (the section headings below), translated naturally into the input's language rather "
     "than left in Korean or English by default. "
     "This document is re-read later as spaced review: it must show clearly both what the user "
     "could explain this week and what they still could not.\n\n"
     "Be noticeably more concise than the daily notes — someone should be able to read only this "
     "document and immediately understand what was learned that week, without opening the daily notes.\n\n"
     "Output format: write real Markdown, not plain numbered lines. Use `##` headings for each "
     "major section listed below, `###` headings for each theme/concept inside the overview "
     "section, `**bold**` for inline emphasis, `-` bullet lists for the still-fuzzy queue, and "
     "`---` horizontal rules between major sections. Never render a toggle, `<details>`, or any "
     "other collapsed/hidden element.\n\n"
     "Output structure, in this exact order:\n"
     "1. A `##` heading (something like '이번 주 학습 정리' in the input's language), followed by a "
     "synthesized overview of the week's concepts and topics as one `###` subsection per theme — "
     "organized by theme rather than by day, merging related concepts that appear across different "
     "days into one entry instead of repeating them. When merging, prefer reusing the user's own "
     "sentences from the daily notes, lightly polished, over paraphrasing them into your own words.\n"
     "2. A `##` heading meaning 'what I still can't explain in my own words' (e.g. Korean "
     "'🌱 아직 내 언어로 안 되는 것'), translated into the input's language: collect every "
     "'🌱 다시 꺼내볼 것' item from the week's daily notes, deduplicated, as a Markdown bullet list "
     "of the still-fuzzy points. Do NOT explain or resolve them — they are the user's retrieval "
     "queue for next week. Omit this section entirely if there are none.\n"
     "3. A final `##` heading (something like '이번 주 배운 것들의 연결' in the input's language) "
     "containing NO bullet points: 2-3 flowing prose paragraphs reading as one continuous narrative "
     "of the week's learning, in the same style as a daily note's closing summary.\n\n"
     "Use only what is explicitly stated in the daily notes provided. Do not invent content, and "
     "do not fill gaps the daily notes marked as unresolved."),
    ("human",
     "Week: {week_start} ~ {week_end}\n\n"
     "This week's daily notes:\n{daily_notes}"),
])


TIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a retrospective-note editor. Turn structured retrospective input into a clean, "
     "generic TIL-style (Today I Learned) retrospective document in Markdown. This must work "
     "equally well for a personal project, a team project, a single day's retrospective, or a dev "
     "journal entry — do not assume any specific category. Match the language of the input — this "
     "includes every section heading below, translated naturally into the input's language rather "
     "than left in English or Korean by default.\n\n"
     "Output format: write real Markdown, not plain numbered lines. Use a `##` heading for each "
     "numbered section below (translate the English heading text into the input's language), "
     "`-` bullet lists where the content is itself a list (e.g. keywords), and `---` horizontal "
     "rules between sections. Never render a toggle, `<details>`, or any other collapsed/hidden "
     "element.\n\n"
     "Output structure, each as its own `##` heading, in this exact order:\n"
     "1. **What happened** — a brief account of the situation or work, from the input.\n"
     "2. **What I learned** — the core takeaway.\n"
     "3. **What I got stuck on** — the problem and how it was resolved. Omit this section if that "
     "input is empty.\n"
     "4. **Reflection** — the reflection input, with grammar and flow polished only. Do NOT expand "
     "a terse reflection into thoughts or sentiments the user did not express — a short honest "
     "reflection is better than a padded one.\n"
     "5. **Next action** — the action-plan input. Omit this section if that input is empty.\n"
     "6. A short keyword list at the end, as a single Markdown bullet list or inline "
     "backtick-tags — not plain comma-separated text.\n\n"
     "Reading this back later should return the user to their own thinking at the time, not to a "
     "smoothed-over version of it. Use only what is explicitly given — an empty or thin section is "
     "acceptable; invented content is not."),
    ("human",
     "What happened: {what}\n"
     "Learned: {learned}\n"
     "Troubleshooting: {troubleshooting}\n"
     "Reflection: {reflection}\n"
     "Next action: {actionplan}\n"
     "Keywords: {keywords}"),
])


BASE_DIR = WRITER_DIR


def slugify(text: str, max_len: int = 20) -> str:
    slug = re.sub(r"\s+", "-", text.strip())
    slug = re.sub(r"[^\w\-가-힣]", "", slug)
    return slug[:max_len].strip("-") or "untitled"


def _read_frontmatter(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    _, frontmatter, _ = raw.split("---", 2)
    return yaml.safe_load(frontmatter)


def save_docs(dir, filename, entry, body, title, overwrite: bool = False) -> Path:
    frontmatter = yaml.safe_dump(entry.model_dump(mode="json"), allow_unicode=True)
    content = f"---\n{frontmatter}---\n\n{body}\n"

    out_path = BASE_DIR / dir / f"{filename}.md"
    if not overwrite:
        counter = 2
        while out_path.exists():
            out_path = BASE_DIR / dir / f"{filename}-{counter}.md"
            counter += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    catalog.upsert_document(out_path, source_type="writer", doc_type=dir, title=title)
    return out_path


def save_raw_session(topic: str, learned: str, related_concepts: list[str]) -> Path:
    today = _date.today()
    slug = slugify(topic)
    out_path = BASE_DIR / "dailynote" / "raw" / f"{today}-{slug}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"topic": topic, "learned": learned, "related_concepts": related_concepts}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_daily_note(
        topic: str,
        learned: str,
        related_concepts: list[str] | None
) -> Path:
    today = _date.today()
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

    response = default_llm().invoke(messages)

    slug = slugify(topic)
    return save_docs("dailynote", f"{today}-{slug}", entry, response.content, title=topic)


def write_weekly_note(as_of: str | None = None) -> Path | None:
    ref_date = _date.fromisoformat(as_of) if as_of else _date.today()
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
        return None

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

    response = default_llm().invoke(messages)

    title = f"{monday.isoformat()} ~ {sunday.isoformat()} 주간노트"
    return save_docs("weeklynote", monday.isoformat(), entry, response.content, title=title, overwrite=True)


def write_tilnote(
    what: str,
    learned: str,
    troubleshooting: str,
    reflection: str,
    actionplan: str,
    keywords: list[str],
) -> Path:
    today = _date.today()
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

    response = default_llm().invoke(messages)

    slug = slugify(what)
    title = what if len(what) <= 60 else f"{what[:60]}…"
    return save_docs("til", f"{today}-{slug}", entry, response.content, title=title)
