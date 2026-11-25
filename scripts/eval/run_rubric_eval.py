"""루브릭 judge를 사람 라벨(rubric_labels.json, 52건)과 비교한다 — 폴백 없이 모델 단독.

사용법:
  uv run python scripts/eval/run_rubric_eval.py exaone [K]        # K = 케이스당 반복(기본 2)
  uv run python scripts/eval/run_rubric_eval.py openrouter [K]    # OPENROUTER_MODEL (무료 50회/일 주의)
  uv run python scripts/eval/run_rubric_eval.py --selftest

측정:
  - 태도 정확도(verdict) + confusion — 기존 judge 기준선(EXAONE 46/49 = 93.9%)과 직접 비교 가능
  - 통과 일치: 라벨 깊이합 >= PASS_DEPTH 인지 vs judge passed
  - 항목별 일치: 정확히 같음 / ±1 이내 / 평균 편향(judge - 라벨)
  - 일관성(K회 모두 같은 verdict), 지연

중단돼도 같은 명령으로 재실행하면 끝난 케이스는 건너뛴다(결과 json에 누적).
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.rag.chain import DIMS, PASS_DEPTH, decide  # noqa: E402
from app.rag.prompts import TURN_PROMPT  # noqa: E402
from app.schemas.rag import RubricResult  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval" / "retrieval_loop"
OUT_DIR = EVAL_DIR / "_workspace"
VERDICTS = ("explained", "partial", "skip")


def make_llm(name: str):
    from langchain_openai import ChatOpenAI
    if name == "exaone":
        return ChatOpenAI(model=os.getenv("EXAONE_MODEL", "k-exaone_v2.5"), base_url="https://api.exaone.ai/v1",
                          api_key=os.getenv("EXAONE_API_KEY"), reasoning_effort="low", max_retries=0)
    if name == "openrouter":
        return ChatOpenAI(model=os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"),
                          base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"),
                          max_retries=0, timeout=90, extra_body={"reasoning": {"enabled": False}})  # 앱과 동일
    sys.exit(f"알 수 없는 모델: {name} (exaone | openrouter)")


def observe(judge, case) -> RubricResult:
    msgs = TURN_PROMPT.invoke({"topic": case["topic"], "next_topic": "none", "message": case["message"],
                               "explanation": case["message"], "conversation": ""}).to_messages()
    return judge.invoke(msgs)


def run(name: str, k: int):
    cases = json.loads((EVAL_DIR / "rubric_labels.json").read_text(encoding="utf-8"))["cases"]
    judge = make_llm(name).with_structured_output(RubricResult)
    out_json = OUT_DIR / f"rubric_eval_runs_{name}.json"
    runs = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    for i, c in enumerate(cases, 1):
        if c["id"] in runs:
            continue
        res = []
        for _ in range(k):
            for attempt in range(3):
                t = time.perf_counter()
                try:
                    r = observe(judge, c)
                    res.append({"sec": round(time.perf_counter() - t, 1), "obs": r.model_dump()})
                    break
                except Exception as e:
                    print(f"  {c['id']} 실패 {attempt + 1}: {type(e).__name__}: {str(e)[:100]}", flush=True)
                    time.sleep(10 * (attempt + 1))
            else:
                res.append({"sec": None, "obs": None})
        runs[c["id"]] = res
        out_json.write_text(json.dumps(runs, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(cases)}] {c['id']} 사람={c['verdict']} judge={[summarize(x)[0] for x in res]}", flush=True)
    path = OUT_DIR / f"rubric_eval_report_{name}.md"
    path.write_text(report(name, k, cases, runs), encoding="utf-8")
    print(f"리포트: {path}")


def summarize(run_):
    """한 회차 → (verdict, passed, 항목 점수 dict). 실패 회차면 (None, None, None)."""
    if not run_ or not run_["obs"]:
        return None, None, None
    r = RubricResult(**run_["obs"])
    # followups를 크게 줘 꼬리질문 분기를 막고, 이 발화 하나로 내린 최종 판정을 본다.
    t = decide(r, followups=10**6, has_next=False)
    return t.verdict, t.passed, {d: getattr(r, d).score for d in DIMS}


def report(name, k, cases, runs) -> str:
    firsts = {c["id"]: next((summarize(x) for x in runs.get(c["id"], []) if x["obs"]), (None, None, None))
              for c in cases}
    scored = [c for c in cases if firsts[c["id"]][0]]
    mx = Counter((c["verdict"], firsts[c["id"]][0]) for c in scored)
    ok = sum(n for (h, j), n in mx.items() if h == j)

    label_pass = {c["id"]: c["verdict"] != "skip" and sum(c["depth"].values()) >= PASS_DEPTH for c in cases}
    pass_ok = sum(1 for c in scored if bool(firsts[c["id"]][1]) == label_pass[c["id"]])

    non_skip = [c for c in scored if c["verdict"] != "skip"]
    dim_rows = []
    for d in DIMS:
        diffs = [firsts[c["id"]][2][d] - c["depth"][d] for c in non_skip]
        dim_rows.append((d, sum(x == 0 for x in diffs), sum(abs(x) <= 1 for x in diffs),
                         sum(diffs) / len(diffs) if diffs else 0.0, len(diffs)))

    rep = [c for c in cases if sum(1 for x in runs.get(c["id"], []) if x["obs"]) >= 2]
    flaky = [c for c in rep if len({summarize(x)[0] for x in runs[c["id"]] if x["obs"]}) > 1]
    secs = [x["sec"] for c in cases for x in runs.get(c["id"], []) if x["sec"]]

    L = [f"# 루브릭 judge 평가 ({name})", "",
         f"- 라벨 {len(cases)}건(rubric_labels.json), 케이스당 {k}회, 폴백 없음, PASS_DEPTH={PASS_DEPTH}",
         f"- 실패로 제외 {len(cases) - len(scored)}건",
         f"- 지연 평균 {sum(secs) / len(secs):.1f}s, 최대 {max(secs):.1f}s" if secs else "- 지연 없음", "",
         f"## 태도 정확도 **{ok}/{len(scored)} = {ok / len(scored):.1%}** (기준선: 기존 judge EXAONE 46/49 = 93.9%)", "",
         "| 사람\\judge | " + " | ".join(VERDICTS) + " |", "|---|---|---|---|"]
    for h in VERDICTS:
        L.append(f"| **{h}** | " + " | ".join(str(mx.get((h, j), 0)) for j in VERDICTS) + " |")
    L += ["", f"## 통과 일치 **{pass_ok}/{len(scored)} = {pass_ok / len(scored):.1%}** "
              f"(라벨 깊이합 >= {PASS_DEPTH} vs judge passed)", "",
          "## 항목별 일치 (skip 제외)", "", "| 항목 | 정확히 일치 | ±1 이내 | 평균 편향(judge-라벨) |", "|---|---|---|---|"]
    for d, exact, near, bias, n in dim_rows:
        L.append(f"| {d} | {exact}/{n} = {exact / n:.0%} | {near}/{n} = {near / n:.0%} | {bias:+.2f} |")
    L += ["", f"## 일관성 {len(rep) - len(flaky)}/{len(rep)} 케이스가 {k}회 모두 같은 verdict", "",
          "## 태도 오판", "", "| id | 사람 | judge | 발화 |", "|---|---|---|---|"]
    for c in scored:
        if firsts[c["id"]][0] != c["verdict"]:
            L.append(f"| {c['id']} | {c['verdict']} | {firsts[c['id']][0]} | {c['message'][:50].replace('|', '/')}… |")
    return "\n".join(L) + "\n"


def selftest():
    """가짜 관찰값으로 집계가 맞는지 확인 — LLM 호출 없음."""
    def obs(score, hedged=False, skipped=False):
        dims = {d: {"evidence": "", "score": score} for d in DIMS}
        return {"sec": 1.0, "obs": {"skipped": skipped, "hedged": hedged, "hedge_evidence": "", **dims}}
    cases = [
        {"id": "a", "message": "m", "verdict": "explained", "depth": dict.fromkeys(DIMS, 1)},
        {"id": "b", "message": "m", "verdict": "partial", "depth": dict.fromkeys(DIMS, 2)},
        {"id": "c", "message": "m", "verdict": "skip", "depth": dict.fromkeys(DIMS, 0)},
    ]
    runs = {"a": [obs(1), obs(1)], "b": [obs(2), obs(2, hedged=True)], "c": [obs(0, skipped=True)]}
    rep = report("selftest", 2, cases, runs)
    assert "2/3 = 66.7%" in rep, rep          # b는 첫 회차가 hedged 아님 → explained로 오판
    assert "통과 일치 **3/3" in rep, rep       # a 깊이 4 통과, b 깊이 8 통과, c skip
    assert "1/2 케이스가 2회 모두" in rep, rep  # b가 흔들림
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
