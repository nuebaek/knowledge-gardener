"""RAG 벤치마크 러너 — 같은 문항으로 retrieval / generation / threshold를 잰다.

    uv run python scripts/benchmark.py --retrieval
    uv run python scripts/benchmark.py --retrieval --embed bge-m3,e5-small-ko-v2
    uv run python scripts/benchmark.py --generation --sample 60
    uv run python scripts/benchmark.py --threshold

검색기는 이름으로 주입한다(RETRIEVERS). 하이브리드(BM25+RRF)를 만들면 함수 하나와
dict 항목 하나를 추가하는 것으로 **같은 문항에서** dense와 나란히 비교된다.
"""
import argparse
import json
import math
import resource
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_chroma import Chroma  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from app.core.paths import PROCESSED_DIR, PROJECT_ROOT  # noqa: E402
from app.rag.chain import (  # noqa: E402
    PROMPT, RELEVANCE_THRESHOLD, TOP_K, apply_fallback, bm25_search, build_bm25, build_chroma_client,
    build_embeddings, build_judge_llm, build_llm, hybrid_merge, load_split_docs, section_key,
)

DATA_PATH = PROJECT_ROOT / "data" / "eval" / "benchmark.jsonl"
RESULTS_DIR = PROJECT_ROOT / "data" / "eval" / "results"

EMBED_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "e5-large": "intfloat/multilingual-e5-large",
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-small": "intfloat/multilingual-e5-small",
    "e5-small-ko-v2": "dragonkue/multilingual-e5-small-ko-v2",
}
ANSWERABLE = ("lexical", "conceptual", "multi_doc")


# ---------------------------------------------------------------- 코퍼스/색인

def public_rows() -> list[dict]:
    """벤치마크는 공개 코퍼스만 쓴다. 서빙 컬렉션(catalog 기반)은 data/writer/를 포함하므로
    그대로 쓰면 gold에 없는 개인 노트가 검색돼 숫자가 오염된다."""
    return [{"source_path": f"data/processed/{p.name}"} for p in sorted(PROCESSED_DIR.glob("*.md"))]


def bench_vectorstore(embeddings, collection: str) -> tuple[Chroma, float]:
    """공개 코퍼스 전용 컬렉션. (vectorstore, 색인에 걸린 초) — 이미 있으면 0.0."""
    client = build_chroma_client()
    existing = {c.name for c in client.list_collections()}
    if collection in existing:
        if client.get_collection(collection).count() > 0:
            return Chroma(client=client, collection_name=collection,
                          embedding_function=embeddings), 0.0
        client.delete_collection(collection)
    started = time.perf_counter()
    store = Chroma.from_documents(load_split_docs(public_rows()), embeddings,
                                  client=client, collection_name=collection)
    return store, time.perf_counter() - started


def make_dense(store):
    return lambda q, k: store.similarity_search(q, k=k)


def make_sparse(store):
    """BM25 only — store는 안 쓰지만 다른 RETRIEVERS 항목과 시그니처를 맞추려 받는다."""
    docs = load_split_docs(public_rows())
    bm25 = build_bm25(docs)
    return lambda q, k: bm25_search(bm25, docs, q, k)


def make_hybrid(store, rrf_k: int = 60):
    """dense + BM25(RRF). BM25는 store와 같은 공개 코퍼스로 만든다 — gold 라벨과 어긋나지
    않게 public_rows()로 고정."""
    docs = load_split_docs(public_rows())
    bm25 = build_bm25(docs)
    dense = make_dense(store)

    def retrieve(q, k):
        return hybrid_merge(dense(q, k), bm25_search(bm25, docs, q, k), k, rrf_k=rrf_k)

    return retrieve


RETRIEVERS = {"dense": make_dense, "sparse": make_sparse, "hybrid": make_hybrid}


# ---------------------------------------------------------------- 지표

def _dedupe(values: list[str]) -> list[str]:
    """순서 보존 중복 제거. 한 섹션에서 여러 청크가 걸려도 순위 1개로 센다."""
    seen, out = set(), []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _ndcg(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, v in enumerate(ranked[:k], 1) if v in gold)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg else 0.0


def score_one(ranked: list[str], gold: set[str], k: int) -> dict:
    top = ranked[:k]
    found = gold & set(top)
    rr = next((1 / i for i, v in enumerate(top, 1) if v in gold), 0.0)
    return {
        "hit": float(bool(found)),
        "recall": len(found) / len(gold),
        "mrr": rr,
        "ndcg": _ndcg(top, gold, k),
    }


def _mean(rows: list[dict], field: str) -> float:
    return statistics.mean(r[field] for r in rows) if rows else 0.0


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1_048_576 if sys.platform == "darwin" else rss / 1024  # darwin=bytes, linux=KB


# ---------------------------------------------------------------- retrieval

def run_retrieval(items: list[dict], retrieve, k: int) -> dict:
    """섹션·문서 두 레벨을 같은 검색 1회에서 파생한다."""
    per_item, latencies = [], []
    for item in items:
        started = time.perf_counter()
        docs = retrieve(item["question"], k * 3)  # 청크→섹션 중복 제거 후 k개를 채우려면 여유가 필요
        latencies.append((time.perf_counter() - started) * 1000)

        sections = _dedupe([section_key(d.metadata) for d in docs])
        documents = _dedupe([d.metadata.get("source", "?") for d in docs])
        gold_sections = {f"{g['doc']}#{g['section']}" for g in item["gold"]}
        gold_docs = {g["doc"] for g in item["gold"]}

        per_item.append({
            "id": item["id"], "type": item["type"], "lang": item["lang"],
            **{f"sec_{m}": v for m, v in score_one(sections, gold_sections, k).items()},
            **{f"doc_{m}": v for m, v in score_one(documents, gold_docs, k).items()},
        })

    def agg(rows):
        return {"n": len(rows), **{f: round(_mean(rows, f), 4) for f in (
            "sec_hit", "sec_recall", "sec_mrr", "sec_ndcg",
            "doc_hit", "doc_recall", "doc_mrr", "doc_ndcg")}}

    by_type = {t: agg([r for r in per_item if r["type"] == t])
               for t in sorted({r["type"] for r in per_item})}
    by_lang = {lg: agg([r for r in per_item if r["lang"] == lg])
               for lg in sorted({r["lang"] for r in per_item})}
    return {
        "overall": agg(per_item), "by_type": by_type, "by_lang": by_lang,
        "latency_ms": {
            "p50": round(statistics.median(latencies), 1),
            "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1),
        },
        "per_item": per_item,
    }


def print_retrieval(label: str, res: dict, k: int) -> None:
    o = res["overall"]
    print(f"\n### {label}  (n={o['n']}, k={k})")
    print(f"| {'slice':<16} | {'n':>3} | {'sec hit':>7} | {'sec MRR':>7} | {'sec nDCG':>8} "
          f"| {'doc hit':>7} | {'doc MRR':>7} |")
    print(f"|{'-' * 18}|{'-' * 5}|{'-' * 9}|{'-' * 9}|{'-' * 10}|{'-' * 9}|{'-' * 9}|")
    rows = [("OVERALL", o)] + list(res["by_type"].items()) + list(res["by_lang"].items())
    for name, m in rows:
        print(f"| {name:<16} | {m['n']:>3} | {m['sec_hit']:>7.0%} | {m['sec_mrr']:>7.3f} "
              f"| {m['sec_ndcg']:>8.3f} | {m['doc_hit']:>7.0%} | {m['doc_mrr']:>7.3f} |")
    lat = res["latency_ms"]
    print(f"latency p50={lat['p50']}ms p95={lat['p95']}ms  "
          f"index={res.get('index_seconds', 0):.1f}s  peak RSS={res.get('peak_rss_mb', 0):.0f}MB")
    if res["by_type"].get("lexical") and res["by_type"].get("conceptual"):
        lex, con = res["by_type"]["lexical"]["sec_mrr"], res["by_type"]["conceptual"]["sec_mrr"]
        print(f"※ lexical MRR {lex:.3f} vs conceptual {con:.3f} — 차이가 거의 없으면 "
              "conceptual 문항이 헤더 어휘를 못 피한 것(데이터셋 문제)")


# ---------------------------------------------------------------- generation

class JudgeVerdict(BaseModel):
    faithful: bool
    correct: bool
    language_match: bool
    refused: bool


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You grade one answer from a document-grounded Q&A system. Return four booleans:\n"
     "- `faithful`: every claim in the answer is supported by the RETRIEVED CONTEXT. "
     "An answer that is correct but not grounded in the context is NOT faithful.\n"
     "- `correct`: the answer matches the REFERENCE. If REFERENCE is empty, judge whether the "
     "answer is a defensible response to the question given the context.\n"
     "- `language_match`: the answer is written in the same language as the question.\n"
     "- `refused`: the answer declines to answer for lack of material "
     "(e.g. '제공된 자료에 이 내용이 없습니다', 'The provided materials do not cover this').\n"
     "Judge only what is written. Do not use outside knowledge."),
    ("human",
     "QUESTION: {question}\n\nREFERENCE: {reference}\n\n"
     "RETRIEVED CONTEXT\n---\n{context}\n---\n\nANSWER: {answer}"),
])


def run_generation(items: list[dict], retrieve, k: int) -> dict:
    from app.rag.chain import _format_docs

    llm = apply_fallback(build_llm())
    judge = JUDGE_PROMPT | build_judge_llm().with_structured_output(JudgeVerdict)
    generate = PROMPT | llm | StrOutputParser()

    rows, latencies = [], []
    for item in items:
        docs = retrieve(item["question"], k)
        context = _format_docs(docs)
        started = time.perf_counter()
        answer = generate.invoke({"context": context, "question": item["question"]})
        latencies.append((time.perf_counter() - started) * 1000)
        v = judge.invoke({"question": item["question"], "reference": item["reference_answer"],
                          "context": context, "answer": answer})
        rows.append({"id": item["id"], "type": item["type"], "lang": item["lang"],
                     **v.model_dump()})

    answerable = [r for r in rows if r["type"] in ANSWERABLE]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    return {
        "n": len(rows),
        "faithfulness": round(_mean(answerable, "faithful"), 4),
        "correctness": round(_mean(answerable, "correct"), 4),
        "language_match": round(_mean(rows, "language_match"), 4),
        # 코퍼스에 답이 없을 때 정직하게 거부했나. 낮으면 없는 얘기를 지어내고 있다는 뜻.
        "refusal_accuracy": round(_mean(unanswerable, "refused"), 4),
        "false_refusal": round(_mean(answerable, "refused"), 4),
        "latency_ms": {"p50": round(statistics.median(latencies), 1),
                       "p95": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1)},
        "per_item": rows,
    }


def print_generation(res: dict) -> None:
    print(f"\n### generation  (n={res['n']})")
    for label, key in [("faithfulness (근거 이탈 안 함)", "faithfulness"),
                       ("correctness", "correctness"),
                       ("language match", "language_match"),
                       ("refusal accuracy (없으면 없다고)", "refusal_accuracy"),
                       ("false refusal (있는데 거부)", "false_refusal")]:
        print(f"  {label:<32} {res[key]:>6.0%}")
    print(f"  latency p50={res['latency_ms']['p50']}ms p95={res['latency_ms']['p95']}ms")


# ---------------------------------------------------------------- threshold

def run_threshold(items: list[dict], store, k: int) -> dict:
    """RELEVANCE_THRESHOLD 캘리브레이션. 양성(정답 문서를 top-1로 맞힌 경우)과
    음성(코퍼스 밖 질문)의 top-1 점수 분포를 비교한다."""
    pos, neg = [], []
    for item in items:
        hits = store.similarity_search_with_relevance_scores(item["question"], k=1)
        if not hits:
            continue
        doc, score = hits[0]
        if item["type"] == "unanswerable":
            neg.append(score)
        elif doc.metadata.get("source") in {g["doc"] for g in item["gold"]}:
            pos.append(score)

    def stats(scores):
        if not scores:
            return {}
        return {"n": len(scores), "mean": round(statistics.mean(scores), 3),
                "median": round(statistics.median(scores), 3),
                "min": round(min(scores), 3), "max": round(max(scores), 3),
                "stdev": round(statistics.stdev(scores), 3) if len(scores) > 1 else 0.0}

    sweep = [{"threshold": round(t / 20, 2),
              "recall": round(sum(s > t / 20 for s in pos) / len(pos), 3) if pos else 0.0,
              "false_accept": round(sum(s > t / 20 for s in neg) / len(neg), 3) if neg else 0.0}
             for t in range(1, 19)]
    return {"positive": stats(pos), "negative": stats(neg), "sweep": sweep,
            "current": RELEVANCE_THRESHOLD}


def print_threshold(res: dict) -> None:
    print(f"\n### threshold  (현재 RELEVANCE_THRESHOLD = {res['current']})")
    print(f"  positive (top-1이 정답 문서): {res['positive']}")
    print(f"  negative (코퍼스 밖 질문)   : {res['negative']}")
    print(f"\n  {'threshold':>9}  {'recall':>7}  {'false-accept':>13}")
    for row in res["sweep"]:
        mark = "  <- 현재" if abs(row["threshold"] - res["current"]) < 0.025 else ""
        print(f"  {row['threshold']:>9.2f}  {row['recall']:>6.0%}  {row['false_accept']:>12.0%}{mark}")
    print("\n  recall을 최대한 지키면서 false-accept이 꺾이는 지점을 고른다.")
    print("  두 분포가 크게 겹치면 top_score 단일 신호로는 못 가른다 — 하이브리드를 붙여도 재조정이 또 필요하다.")


# ---------------------------------------------------------------- 실행

def load_items(sample: int | None, seed: int = 20260731) -> list[dict]:
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH.relative_to(PROJECT_ROOT)}가 없습니다. "
                 "먼저 `uv run python scripts/gen_evalset.py`를 실행하세요.")
    items = [json.loads(line) for line in DATA_PATH.read_text(encoding="utf-8").splitlines() if line]
    if sample and sample < len(items):
        import random
        # 타입별 층화 — 무작위로 뽑으면 unanswerable이 통째로 빠질 수 있다.
        rng, buckets = random.Random(seed), defaultdict(list)
        for item in items:
            buckets[item["type"]].append(item)
        out = []
        for bucket in buckets.values():
            rng.shuffle(bucket)
            out += bucket[: max(1, round(sample * len(bucket) / len(items)))]
        return out
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--threshold", action="store_true")
    ap.add_argument("--embed", default="bge-m3", help=f"쉼표 구분. 가능: {','.join(EMBED_MODELS)}")
    ap.add_argument("--retriever", default="dense", help=f"가능: {','.join(RETRIEVERS)}")
    ap.add_argument("--k", type=int, default=TOP_K)
    ap.add_argument("--rrf-k", type=int, default=60, help="hybrid 리트리버 전용 RRF 상수")
    ap.add_argument("--sample", type=int, help="문항 수 제한 (타입별 층화)")
    args = ap.parse_args()

    if not (args.retrieval or args.generation or args.threshold):
        ap.error("--retrieval / --generation / --threshold 중 최소 하나가 필요합니다.")

    items = load_items(args.sample)
    answerable = [i for i in items if i["type"] in ANSWERABLE]
    print(f"문항 {len(items)}개 (answerable {len(answerable)}, "
          f"unanswerable {len(items) - len(answerable)})")

    report = {"at": datetime.now(timezone.utc).isoformat(), "k": args.k,
              "n_items": len(items), "runs": {}}

    for name in args.embed.split(","):
        name = name.strip()
        if name not in EMBED_MODELS:
            sys.exit(f"알 수 없는 임베딩: {name}. 가능: {', '.join(EMBED_MODELS)}")
        build_embeddings.cache_clear()  # maxsize=1 — 교체 시 이전 모델을 해제해야 메모리가 안 샌다
        store, index_seconds = bench_vectorstore(
            build_embeddings(EMBED_MODELS[name]), f"bench_{name.replace('-', '_')}")
        retrieve = (make_hybrid(store, rrf_k=args.rrf_k) if args.retriever == "hybrid"
                    else RETRIEVERS[args.retriever](store))
        run = {}

        if args.retrieval:
            res = run_retrieval(answerable, retrieve, args.k)
            res["index_seconds"] = round(index_seconds, 2)
            res["peak_rss_mb"] = round(_peak_rss_mb(), 1)
            print_retrieval(f"{name} / {args.retriever}", res, args.k)
            run["retrieval"] = res
        if args.generation:
            res = run_generation(items, retrieve, args.k)
            print_generation(res)
            run["generation"] = res
        if args.threshold:
            res = run_threshold(items, store, args.k)
            print_threshold(res)
            run["threshold"] = res

        report["runs"][f"{name}/{args.retriever}"] = run

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.retriever}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
