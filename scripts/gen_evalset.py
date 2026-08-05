"""공개 코퍼스(data/processed/)에서 RAG 벤치마크 문항을 생성·검증한다.

gold는 섹션 단위(`data/processed/rnn.md#RNN > LSTM`)라 chunk_size를 바꿔도 라벨이 안 깨진다.
data/writer/(개인 노트)는 절대 건드리지 않는다 — 남이 clone해서 같은 숫자를 못 뽑으면 벤치마크가 아니다.

    uv run python scripts/gen_evalset.py --probe        # 5섹션 실측 → 전체 비용 추정만
    uv run python scripts/gen_evalset.py                # 전체 생성 (비용 확인 후 진행)
    uv run python scripts/gen_evalset.py --review 25    # 생성된 문항을 gold 원문과 대조 출력
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.core.paths import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from app.rag.chain import build_judge_llm, load_split_docs, section_key  # noqa: E402

OUT_PATH = PROJECT_ROOT / "data" / "eval" / "benchmark.jsonl"

GEN_MODELS = ("anthropic", "google", "cerebras")   # 생성기 모델. 데이터셋 provenance로 각 문항에 기록된다

MIN_SECTION_CHARS = 300      # 이보다 짧은 섹션은 문항을 만들 내용이 없다
HEADER_OVERLAP_MAX = 0.4     # conceptual 문항이 헤더 어휘를 이 비율 이상 재사용하면 탈락
QUOTA = {"en": 100, "ko": 100}
UNANSWERABLE_TARGET = 40
MAX_CONCURRENCY = 4
SEED = 20260731              # 샘플링 재현성

# Sonnet 5 인트로 가격 (2026-08-31까지, 이후 $3/$15)
PRICE_IN, PRICE_OUT = 2.0 / 1_000_000, 10.0 / 1_000_000


# ---------------------------------------------------------------- 코퍼스

def public_sections() -> dict[str, str]:
    """공개 코퍼스를 섹션 단위로 묶는다. {section_key: 본문}"""
    rows = [
        {"source_path": f"data/processed/{p.name}"}
        for p in sorted(PROCESSED_DIR.glob("*.md"))
    ]
    merged: dict[str, list[str]] = defaultdict(list)
    for chunk in load_split_docs(rows):
        merged[section_key(chunk.metadata)].append(chunk.page_content)
    sections = {k: "\n\n".join(v) for k, v in merged.items()}
    return {k: t for k, t in sections.items() if len(t) >= MIN_SECTION_CHARS}


_HANGUL = re.compile(r"[가-힣]")


def lang_of(text: str) -> str:
    """한글 비율로 판정. 파일명 규칙(week-*-TIL)보다 문서가 섞여도 안전하다."""
    return "ko" if len(_HANGUL.findall(text)) / max(len(text), 1) > 0.05 else "en"


# ------------------------------------------------- 헤더 어휘 겹침 (LLM 아님)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|[가-힣]+")
# 조사/어미를 붙인 채 비교하면 "LSTM은" vs "LSTM"이 다른 단어가 된다. 형태소 분석기는 안 쓴다.
# ponytail: 접미사 스트립 근사. 오판이 눈에 띄면 그때 kiwipiepy 같은 걸 붙인다.
_KO_SUFFIX = re.compile(r"(으로서|으로써|에서는|에게|에서|으로|이랑|까지|부터|보다|처럼|만큼"
                        r"|은|는|이|가|을|를|의|에|와|과|도|만|나|로|랑|함|들)$")
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "for", "is", "are", "with",
         "그리고", "또는", "위한", "대한", "있는", "하는"}


def content_words(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(text.lower()):
        if _HANGUL.match(w) and len(w) > 2:
            w = _KO_SUFFIX.sub("", w) or w
        if len(w) > 1 and w not in _STOP:
            out.add(w)
    return out


def header_overlap(question: str, key: str) -> float:
    """질문이 섹션 헤더 어휘를 얼마나 그대로 베꼈나. 1.0이면 제목 패러프레이즈."""
    header = content_words(key.split("#", 1)[-1])
    if not header:
        return 0.0
    return len(header & content_words(question)) / len(header)


# ---------------------------------------------------------------- 스키마

class QAPair(BaseModel):
    lexical_question: str = Field(description="Uses the section's exact technical terms.")
    lexical_answer: str
    conceptual_question: str = Field(
        description="Asks the same underlying concept while AVOIDING the section heading's wording."
    )
    conceptual_answer: str


class Verdict(BaseModel):
    lexical_grounded: bool
    lexical_self_contained: bool
    conceptual_grounded: bool
    conceptual_self_contained: bool
    note: str = ""


class SectionPick(BaseModel):
    sections: list[str] = Field(description="Verbatim section headings that answer the question.")


class Unanswerable(BaseModel):
    questions: list[str]


# ---------------------------------------------------------------- 프롬프트

GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write retrieval-benchmark questions from one section of a document corpus.\n\n"
     "Write TWO questions answerable from this section ALONE, plus a reference answer for each:\n"
     "1. `lexical` — uses the section's own precise technical terms. A keyword search should find it.\n"
     "2. `conceptual` — asks the same underlying idea while DELIBERATELY AVOIDING the words in the "
     "section heading. Describe the mechanism or its purpose instead of naming it. This one exists "
     "to test semantic search, so reusing heading vocabulary makes it worthless.\n\n"
     "HARD RULES:\n"
     "- Every question must stand alone. No 'this document', 'the above', 'as mentioned', and no "
     "pronoun whose referent is only in the section.\n"
     "- Reference answers must be fully supported by the section text. Never add outside knowledge.\n"
     "- Write in {language}. (ko = 한국어, en = English.)"),
    ("human", "Section heading: {heading}\n\n---\n{body}\n---"),
])

VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You audit benchmark questions against their source section. Judge each question twice:\n"
     "- `grounded`: is the reference answer fully supported by the section text below? "
     "If it needs outside knowledge or states something the section does not, this is false.\n"
     "- `self_contained`: can the question be understood by someone who has never seen this "
     "section? Any 'this document' / 'the above' / dangling pronoun makes it false.\n\n"
     "Be strict — a bad item silently corrupts every measurement taken with it."),
    ("human",
     "SECTION\n---\n{body}\n---\n\n"
     "LEXICAL Q: {lexical_question}\nLEXICAL A: {lexical_answer}\n\n"
     "CONCEPTUAL Q: {conceptual_question}\nCONCEPTUAL A: {conceptual_answer}"),
])

PICK_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given a question and the section headings of the document that answers it, return the "
     "heading(s) whose content actually answers the question. Copy headings VERBATIM from the list. "
     "Prefer one; return two only if the answer genuinely spans both."),
    ("human", "QUESTION: {question}\n\nHEADINGS IN {doc}:\n{headings}"),
])

UNANSWERABLE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You write NEGATIVE test questions for a retrieval benchmark. Given the full table of contents "
     "of a corpus, write {n} questions that a user of this system might plausibly ask but that the "
     "corpus CANNOT answer — adjacent topics that are genuinely absent, not absurd ones. "
     "They must be hard negatives: same broad field, wrong specific topic. "
     "Write them in {language}. Return questions only."),
    ("human", "TABLE OF CONTENTS\n{toc}"),
])


# ---------------------------------------------------------------- 생성

def gen_llm(name: str):
    """생성·검증용 LLM. 벤치마크 데이터셋의 품질이 여기서 결정되므로 어떤 모델로 만들었는지를
    각 문항에 기록한다(`model` 필드) — 공개 데이터셋에서 provenance는 필수 정보다."""
    if name == "anthropic":
        return build_judge_llm()
    if name == "google":
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_google_genai import ChatGoogleGenerativeAI

        # ponytail: 무료 티어 분당 5요청 고정 스로틀. 유료 등급으로 올리면 지워도 됨.
        limiter = InMemoryRateLimiter(requests_per_second=5 / 65, max_bucket_size=1)
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            rate_limiter=limiter,
            max_retries=3,
        )
    if name == "cerebras":
        from langchain_cerebras import ChatCerebras
        return ChatCerebras(
            model=os.getenv("CEREBRAS_MODEL", "gemma-4-31b"),
            api_key=os.getenv("CEREBRAS_API_KEY"),
            max_retries=0,
        )
    sys.exit(f"알 수 없는 모델: {name}. 가능: {', '.join(GEN_MODELS)}")


def _usage(responses) -> tuple[int, int]:
    tin = tout = 0
    for r in responses:
        meta = getattr(r, "usage_metadata", None) or {}
        tin += meta.get("input_tokens", 0)
        tout += meta.get("output_tokens", 0)
    return tin, tout


def _batch(chain, inputs):
    return chain.batch(inputs, config={"max_concurrency": MAX_CONCURRENCY})


def generate_items(keys: list[str], sections: dict[str, str], llm,
                   model_name: str = "anthropic") -> tuple[list[dict], dict]:
    """섹션마다 lexical/conceptual 2문항 생성 → 검증 → 어휘 겹침 필터."""
    gen = GEN_PROMPT | llm.with_structured_output(QAPair)
    inputs = [
        {"heading": k.split("#", 1)[-1], "body": sections[k], "language": lang_of(sections[k])}
        for k in keys
    ]
    pairs = _batch(gen, inputs)

    ver = VERIFY_PROMPT | llm.with_structured_output(Verdict)
    verdicts = _batch(ver, [
        {"body": sections[k], **p.model_dump()} for k, p in zip(keys, pairs)
    ])

    items, dropped = [], defaultdict(int)
    for key, pair, v in zip(keys, pairs, verdicts):
        lang = lang_of(sections[key])
        doc, heading = key.split("#", 1)
        for kind in ("lexical", "conceptual"):
            question = getattr(pair, f"{kind}_question")
            if not (getattr(v, f"{kind}_grounded") and getattr(v, f"{kind}_self_contained")):
                dropped[f"{kind}/llm-verify"] += 1
                continue
            if kind == "conceptual" and header_overlap(question, key) >= HEADER_OVERLAP_MAX:
                dropped["conceptual/header-overlap"] += 1
                continue
            items.append({
                "id": f"{Path(doc).stem}-{kind[:3]}-{len(items):03d}",
                "lang": lang,
                "type": kind,
                "question": question,
                "reference_answer": getattr(pair, f"{kind}_answer"),
                "gold": [{"doc": doc, "section": heading}],
                "origin": "generated",
                "model": model_name,
            })
    return items, dict(dropped)


def migrate_human(sections: dict[str, str], llm) -> list[dict]:
    """기존 손으로 쓴 영어 37문항에 섹션 라벨을 붙여 흡수한다.
    한국어 10문항은 gold가 data/writer/(비공개)라 이관 불가 — 폐기."""
    from eval_retrieval import EVAL_SET, HARD_EVAL_SET  # noqa: E402

    by_doc: dict[str, list[str]] = defaultdict(list)
    for key in sections:
        doc, heading = key.split("#", 1)
        by_doc[doc].append(heading)

    tasks = [(q, docs) for q, docs in EVAL_SET + HARD_EVAL_SET
             if all(d in by_doc for d in docs)]
    skipped = len(EVAL_SET) + len(HARD_EVAL_SET) - len(tasks)
    if skipped:
        print(f"  · human 문항 {skipped}개 스킵 (gold 문서에 섹션이 없음 — 본문이 너무 짧음)")

    pick = PICK_PROMPT | llm.with_structured_output(SectionPick)
    # 문항당 gold 문서가 여럿일 수 있어 (문항, 문서) 쌍 단위로 부른다.
    flat = [(i, d) for i, (q, docs) in enumerate(tasks) for d in docs]
    picks = _batch(pick, [
        {"question": tasks[i][0], "doc": d, "headings": "\n".join(by_doc[d])}
        for i, d in flat
    ])

    gold_by_task: dict[int, list[dict]] = defaultdict(list)
    for (i, doc), p in zip(flat, picks):
        for heading in p.sections:
            if heading in by_doc[doc]:
                gold_by_task[i].append({"doc": doc, "section": heading})

    items = []
    for i, (question, docs) in enumerate(tasks):
        gold = gold_by_task.get(i)
        if not gold:
            continue
        items.append({
            "id": f"human-{i:03d}",
            "lang": "en",
            "type": "multi_doc" if len(docs) > 1 else "conceptual",
            "question": question,
            "reference_answer": "",   # 원본에 정답문이 없다. generation 채점에서 제외된다.
            "gold": gold,
            "origin": "human",
        })
    return items


def build_unanswerable(sections: dict[str, str], llm) -> list[dict]:
    """코퍼스가 실제로 다루는 영역 바로 옆에 있지만 색인엔 없는 주제."""
    from eval_retrieval import HARD_NEGATIVE_EVAL_SET, NEGATIVE_EVAL_SET  # noqa: E402

    items = [
        {"id": f"neg-hand-{i:03d}", "lang": "ko", "type": "unanswerable",
         "question": q, "reference_answer": "", "gold": [], "origin": "human"}
        for i, q in enumerate(NEGATIVE_EVAL_SET + HARD_NEGATIVE_EVAL_SET)
    ]

    toc = defaultdict(list)
    for key, body in sections.items():
        toc[lang_of(body)].append(key.split("#", 1)[-1])

    need = UNANSWERABLE_TARGET - len(items)
    chain = UNANSWERABLE_PROMPT | llm.with_structured_output(Unanswerable)
    results = _batch(chain, [
        {"n": need // 2, "language": lang, "toc": "\n".join(sorted(set(toc[lang]))[:120])}
        for lang in ("en", "ko")
    ])
    for lang, res in zip(("en", "ko"), results):
        for q in res.questions:
            items.append({
                "id": f"neg-gen-{len(items):03d}", "lang": lang, "type": "unanswerable",
                "question": q, "reference_answer": "", "gold": [], "origin": "generated",
            })
    return items[:UNANSWERABLE_TARGET]


# ---------------------------------------------------------------- 실행

def sample_keys(sections: dict[str, str], per_lang: dict[str, int]) -> list[str]:
    """언어별 쿼터를 강제한다. 섹션 수로만 뽑으면 영어(cs231n)가 압도해 한국어 슬라이스가 비어버린다."""
    rng = random.Random(SEED)
    buckets = defaultdict(list)
    for key, body in sections.items():
        buckets[lang_of(body)].append(key)
    picked = []
    for lang, n_items in per_lang.items():
        pool = sorted(buckets[lang])
        rng.shuffle(pool)
        # 섹션당 2문항, 검증 탈락 25% 감안
        want = int(n_items / 2 * 1.35) + 1
        if len(pool) < want:
            # 조용히 적게 만들면 "왜 한국어 문항이 78개지?"를 나중에 디버깅하게 된다
            print(f"  ⚠ {lang}: 섹션 {len(pool)}개뿐 (쿼터 {n_items}문항엔 {want}개 필요) "
                  f"→ 최대 ~{int(len(pool) * 2 * 0.75)}문항")
        picked += pool[:want]
    return picked


def review(n: int) -> None:
    sections = public_sections()
    items = [json.loads(line) for line in OUT_PATH.read_text(encoding="utf-8").splitlines()]
    for item in random.Random(SEED).sample(items, min(n, len(items))):
        print(f"\n{'=' * 78}\n[{item['type']}/{item['lang']}/{item['origin']}] {item['id']}")
        print(f"Q: {item['question']}")
        if item["reference_answer"]:
            print(f"A: {item['reference_answer']}")
        for g in item["gold"]:
            key = f"{g['doc']}#{g['section']}"
            body = sections.get(key, "(섹션을 찾을 수 없음 — 라벨 불일치!)")
            print(f"\n  GOLD {key}\n  {body[:600].replace(chr(10), chr(10) + '  ')}…")
        if not item["gold"]:
            print("  GOLD: (없음 — unanswerable)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="5섹션만 돌려 전체 비용을 실측 추정")
    ap.add_argument("--review", type=int, metavar="N", help="생성된 문항 N개를 gold 원문과 대조")
    ap.add_argument("--yes", action="store_true", help="비용 확인 프롬프트 건너뛰기")
    ap.add_argument("--model", default="cerebras", choices=GEN_MODELS,
                    help="생성·검증 모델 (기본 cerebras — anthropic/google은 API 크레딧 별도 과금)")
    ap.add_argument("--sections", action="store_true", help="코퍼스 통계만 출력, LLM 호출 없음")
    args = ap.parse_args()

    if args.review:
        return review(args.review)

    sections = public_sections()
    by_lang = defaultdict(int)
    for body in sections.values():
        by_lang[lang_of(body)] += 1
    print(f"공개 코퍼스: 섹션 {len(sections)}개 (en {by_lang['en']}, ko {by_lang['ko']})"
          f" — {MIN_SECTION_CHARS}자 미만 제외")

    keys = sample_keys(sections, QUOTA)
    if args.sections:
        return print(f"샘플링 대상 {len(keys)}섹션 → LLM {len(keys) * 2 + 30}콜 예상 (호출 안 함)")

    llm = gen_llm(args.model)

    if args.probe:
        probe_keys = sample_keys(sections, {"en": 6, "ko": 6})[:5]
        print(f"\n[probe] {args.model} / {len(probe_keys)}개 섹션…")
        probe_items, dropped = generate_items(probe_keys, sections, llm, args.model)
        print(f"  생성 {len(probe_items)}문항, 탈락 {dropped or '없음'}")
        print(f"  전체 {len(keys)}섹션 기준 예상: 약 {len(keys) * 2}콜")
        print("  ※ 실제 사용량은 LangSmith(kaia 프로젝트) 트레이스에서 확인하세요.")
        return

    if not args.yes:
        print(f"\n전체 실행: {len(keys)}섹션 × 2콜 + human 라벨링 + unanswerable ≈ {len(keys) * 2 + 30}콜")
        cost_note = {
            "anthropic": " — ANTHROPIC_API_KEY 선불 크레딧 과금. Claude Pro/Max 구독으로는 못 씁니다.",
            "google": " — GOOGLE_API_KEY 과금",
            "cerebras": " — CEREBRAS_API_KEY (무료 티어)",
        }[args.model]
        print(f"모델: {args.model}{cost_note}")
        if input("진행할까요? [y/N] ").strip().lower() != "y":
            return print("취소했습니다.")

    print(f"\n[1/3] 문항 생성·검증 — {len(keys)}섹션 ({args.model})")
    items, dropped = generate_items(keys, sections, llm, args.model)
    print(f"  → {len(items)}문항 통과, 탈락 {dropped or '없음'}")

    print("[2/3] 기존 human 문항 섹션 라벨링")
    human = migrate_human(sections, llm)
    print(f"  → {len(human)}문항")

    print("[3/3] unanswerable")
    negs = build_unanswerable(sections, llm)
    print(f"  → {len(negs)}문항")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in items + human + negs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n{OUT_PATH.relative_to(PROJECT_ROOT)} — 총 {len(items) + len(human) + len(negs)}문항")
    print("다음: uv run python scripts/gen_evalset.py --review 25 로 눈으로 검수하세요.")


if __name__ == "__main__":
    main()
