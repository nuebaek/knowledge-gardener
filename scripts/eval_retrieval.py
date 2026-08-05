import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chain import (  # noqa: E402
    RELEVANCE_THRESHOLD, TOP_K, _extract_sources, bm25_search, build_bm25, build_embeddings,
    get_vectorstore, hybrid_merge, load_split_docs, search_source_paths,
)

# --compare용. 모델마다 벡터 차원이 다르므로 컬렉션을 분리해야 한다(섞으면 차원 불일치로 터짐).
COMPARE_MODELS = [
    ("bge-m3 (현재, 568M)", "BAAI/bge-m3", "bench_bge_m3"),
    ("e5-large (560M)", "intfloat/multilingual-e5-large", "bench_e5_large"),
    ("e5-base (278M)", "intfloat/multilingual-e5-base", "bench_e5_base"),
    ("e5-small (118M)", "intfloat/multilingual-e5-small", "bench_e5_small"),
    ("e5-small-ko-v2 (118M, dragonkue)", "dragonkue/multilingual-e5-small-ko-v2", "bench_e5_small_ko"),
]

EVAL_SET = [
    ("What is an adversarial example in the context of image classifiers?",
     ["data/processed/adversary-attacks.md"]),
    ("What problem does an attention mechanism solve in sequence models?",
     ["data/processed/attention.md", "data/processed/transformers.md"]),
    ("What makes image classification a hard problem compared to writing explicit rules?",
     ["data/processed/classification.md"]),
    ("What are common techniques to address overfitting when training a ConvNet?",
     ["data/processed/convnet-tips.md"]),
    ("How does a convolutional layer differ from a fully connected layer?",
     ["data/processed/convolutional-networks.md"]),
    ("How does a Variational Autoencoder learn to generate new data?",
     ["data/processed/generative-modeling.md"]),
    ("What's the difference between PixelCNN and PixelRNN in terms of training speed?",
     ["data/processed/generative-models.md", "data/processed/pixelrnn.md"]),
    ("What is the score function in a linear classifier?",
     ["data/processed/linear-classify.md"]),
    ("What does NeRF stand for and what problem does it solve?",
     ["data/processed/nerf.md"]),
    ("How is a single neuron modeled mathematically in a neural network?",
     ["data/processed/neural-networks-1.md"]),
    ("Why is data preprocessing like whitening useful before training a neural network?",
     ["data/processed/neural-networks-2.md"]),
    ("What is the difference between vanilla gradient descent and momentum update?",
     ["data/processed/neural-networks-3.md"]),
    ("How do you train a softmax linear classifier on toy 2D data?",
     ["data/processed/neural-networks-case-study.md"]),
    ("What is the role of the loss function in optimization for image classification?",
     ["data/processed/optimization-1.md"]),
    ("How does backpropagation compute gradients in a neural network?",
     ["data/processed/optimization-2.md"]),
    ("How does PixelRNN model the probability distribution of image pixels?",
     ["data/processed/pixelrnn.md", "data/processed/generative-models.md"]),
    ("What is an LSTM and why does it help with long-term dependencies?",
     ["data/processed/rnn.md"]),
    ("When is transfer learning a good approach for a new dataset?",
     ["data/processed/transfer-learning.md"]),
    ("What is multi-headed attention in a Transformer?",
     ["data/processed/transformers.md", "data/processed/attention.md"]),
    ("How can we visualize what a ConvNet has learned?",
     ["data/processed/understanding-cnn.md"]),
]

# 위 EVAL_SET은 문서 헤더 어휘를 그대로 따온 질문이 많아, 그냥 제목 패러프레이즈 매칭을
# 재는 걸 수도 있다. 헤더 단어를 일부러 피해서 개념만 묻는 질문으로 숫자가 버티는지 확인.
HARD_EVAL_SET = [
    ("Why can't we just hard-code rules to recognize a cat versus a dog in a photo?",
     ["data/processed/classification.md"]),
    ("What lets a network reuse the same small detector across every position in an "
     "image instead of learning a separate weight per pixel?",
     ["data/processed/convolutional-networks.md"]),
    ("Why might adding a 'velocity' term to gradient descent help it get past shallow "
     "dips in the loss surface?",
     ["data/processed/neural-networks-3.md"]),
    ("Why do plain recurrent networks struggle to remember something from many steps "
     "earlier in a sequence, and what fixes that?",
     ["data/processed/rnn.md"]),
    ("If I only have a small dataset, why might reusing weights from a network trained "
     "on a huge unrelated dataset still help?",
     ["data/processed/transfer-learning.md"]),
    ("If a tiny, humanly-imperceptible change to an image can flip a classifier's "
     "prediction, what does that reveal about the model's decision boundary?",
     ["data/processed/adversary-attacks.md"]),
    ("In a linear scoring approach, how is each class's score computed directly from "
     "the raw pixel values?",
     ["data/processed/linear-classify.md"]),
    ("What is the mathematical role of a single unit that takes weighted inputs and "
     "squashes them through a nonlinearity?",
     ["data/processed/neural-networks-1.md"]),
    ("Why would you center and rescale your input features before feeding them into "
     "a network, rather than using the raw values?",
     ["data/processed/neural-networks-2.md"]),
    ("How would you build a toy 2D dataset to test whether a simple model can separate "
     "classes that aren't linearly separable?",
     ["data/processed/neural-networks-case-study.md"]),
    ("Why can't you just solve directly for the best set of weights instead of "
     "iteratively searching for them?",
     ["data/processed/optimization-1.md"]),
    ("How do you compute the derivative of a chain of operations, like squaring the "
     "sum of two inputs?",
     ["data/processed/optimization-2.md"]),
    ("How do researchers try to peek inside a trained image network to see what "
     "patterns it has actually picked up on?",
     ["data/processed/understanding-cnn.md"]),
    ("How can a handful of 2D photos taken from different angles be turned into "
     "something you can render brand-new viewpoints from?",
     ["data/processed/nerf.md"]),
    ("What kind of generative approach learns a compressed representation it can "
     "sample from to produce new, similar-looking data?",
     ["data/processed/generative-modeling.md"]),
    ("How can you generate an image one pixel at a time, using the pixels already "
     "generated to predict the next one?",
     ["data/processed/pixelrnn.md", "data/processed/generative-models.md"]),
    ("What lets a model weigh how much to focus on different positions in the input "
     "when producing each part of its output?",
     ["data/processed/attention.md", "data/processed/transformers.md"]),
]


# 한국어 코퍼스(TIL/writer) 문항. 위 EVAL_SET/HARD_EVAL_SET은 전부 영어(cs231n)라 이쪽을 안 잰다 —
# 다국어 모델(e5 계열, 특히 한국어 파인튠)을 비교할 땐 이게 없으면 반쪽짜리 비교다.
# 헤더/topic 단어를 피해 개념만 물어서 표면 어휘 매칭이 아닌 진짜 의미 검색을 잰다.
# RAG-Reranking 계열은 2026-07-28에 같은 내용이 5개 파일로 중복 저장된 데이터 품질 이슈가 있어
# 5개 전부를 정답으로 인정한다(아래 _RAG_DUPES).
_RAG_DUPES = [
    "data/writer/dailynote/2026-07-28-RAG-Reranking.md",
    "data/writer/dailynote/2026-07-28-rag-reraking.md",
    "data/writer/dailynote/2026-07-28-RAG-검색-reranking.md",
    "data/writer/dailynote/2026-07-28-RAG와-Reranking.md",
    "data/writer/dailynote/2026-07-28-RAG와-Reranking-2.md",
]

KOREAN_EVAL_SET = [
    ("컨테이너 안에 있던 로그나 DB 파일처럼, 컨테이너를 지우고 다시 만들면 사라지는 데이터를 지키려면 뭘 써야 해?",
     ["data/writer/dailynote/2026-07-22-Docker-및-배포-전략.md"]),
    ("같은 이미지를 여러 서버에 그대로 옮겨서 똑같은 실행 환경을 재현하려면 어떻게 해?",
     ["data/writer/dailynote/2026-07-30-Docker-공부.md", "data/writer/dailynote/2026-07-22-Docker-및-배포-전략.md"]),
    ("이전 단계에서 이미 계산해 둔 값을 매번 새로 구하지 않고 저장해뒀다가 다음 토큰 생성할 때 재사용해서 연산을 아끼는 방식이 뭐야?",
     ["data/writer/dailynote/2026-07-29-KV-Cache-Continuous.md"]),
    ("사전학습된 모델 가중치는 그대로 두고, 추가로 학습 가능한 부분만 얹어서 학습 비용을 줄이는 방법을 뭐라고 해?",
     ["data/writer/dailynote/2026-07-29-모델-학습-기법-Finetuning.md"]),
    ("크고 성능 좋은 모델의 출력을 정답 삼아서 더 작은 모델을 학습시키는 방식은 뭐야?",
     ["data/writer/dailynote/2026-07-29-모델-학습-기법-Finetuning.md"]),
    ("실수형 가중치를 더 작은 정수형으로 바꿔서 모델이 차지하는 용량과 연산량을 줄이는 기법이 뭐야?",
     ["data/writer/dailynote/2026-07-19-LLM-Optimizer.md"]),
    ("검색으로 찾아온 문서 후보들의 순서를 한 번 더 정교하게 매겨서 진짜 관련 있는 걸 앞으로 보내는 단계를 뭐라고 해?",
     _RAG_DUPES),
    ("질문이 들어오면 관련 문서를 먼저 찾고, 그 내용을 근거로 답을 생성하는 방식을 뭐라고 해?",
     _RAG_DUPES),
    ("매번 프로젝트를 새로 시작할 때마다 DB 상태를 수동으로 맞춰야 해서 반복 작업이 되는 문제, 이걸 어떻게 개선할지 고민한 기록이 어디 있어?",
     ["data/writer/til/2026-07-20-디비-관련-프로젝트-업데이트-진행-및.md"]),
    ("API가 초당 얼마나 많은 요청을 처리하는지와, 요청 하나가 응답 오기까지 걸리는 시간, 이 두 가지를 다룬 문서가 뭐야?",
     ["data/writer/dailynote/2026-07-30-성능-지표.md"]),
]


# ---- RELEVANCE_THRESHOLD 캘리브레이션용 음성 예시 ----
# 위 EVAL_SET/HARD_EVAL_SET은 전부 "코퍼스에 답이 있는" 문항이라 hit-rate/MRR은 재도
# top_score 임계값(grade_docs가 "이 정도면 답해도 되나" 가르는 값)은 못 잰다 — 그러려면
# "점수가 낮게 나와야 정상인" 문항도 있어야 두 분포를 비교할 수 있다.
#
# GENERIC: 이 코퍼스 어떤 주제와도 무관한 일반 상식 — 명백한 음성 대조군.
NEGATIVE_EVAL_SET = [
    "김치찌개 맛있게 끓이는 법이 뭐야?",
    "에베레스트산의 높이는 얼마야?",
    "제주도 2박 3일 여행 코스 추천해줘",
    "축구 오프사이드 규칙이 뭐야?",
    "커피 원두 로스팅 단계별 차이가 뭐야?",
    "고양이랑 강아지 중에 뭐가 더 키우기 쉬워?",
    "아인슈타인은 몇 년도에 태어났어?",
    "여름철 냉방병 예방하는 방법 알려줘",
]

# HARD_NEGATIVE: 이 코퍼스가 실제로 다루는 영역(ML/RAG/Docker) 바로 옆에 있지만 색인엔 없는 주제 — "명백히 무관"보다 훨씬 어려운, 실전에 가까운 음성 대조군.
HARD_NEGATIVE_EVAL_SET = [
    "쿠버네티스에서 Pod와 Deployment의 차이가 뭐야?",
    "GraphQL이 REST API보다 나은 점이 뭐야?",
    "PostgreSQL에서 인덱스는 언제 걸어야 해?",
    "git rebase랑 merge는 언제 각각 써야 해?",
    "TCP와 UDP의 차이가 뭐야?",
    "React useEffect 의존성 배열은 왜 필요해?",
]


def _reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    """정답이 처음 나오는 순위의 역수. top-k 안에 없으면 0. MRR의 문항별 값."""
    for i, path in enumerate(retrieved, start=1):
        if path in gold:
            return 1.0 / i
    return 0.0


def _run(label: str, eval_set: list, retriever) -> dict:
    """검색기(retriever: query,k -> [path])를 주입받아 문항별로 3지표 계산"""
    hits = recall_sum = prec_sum = rr_sum = 0.0
    for question, expected in eval_set:
        gold = set(expected)
        retrieved = retriever(question, k=TOP_K)
        topk = retrieved[:TOP_K]
        found = gold & set(topk)
        hit = bool(found)
        rr = _reciprocal_rank(topk, gold)
        hits += hit
        recall_sum += len(found) / len(gold)
        prec_sum += len(found) / len(topk) if topk else 0.0
        rr_sum += rr
        rank = next((i for i, p in enumerate(topk, 1) if p in gold), None)
        print(f"[{'OK  ' if hit else 'MISS'}] rank={rank or '-'}  {question}")
        print(f"       expected={expected}")
        print(f"       retrieved={retrieved}")
    n = len(eval_set)
    print(f"\n{label}: hit-rate@{TOP_K}={hits/n:.0%}  recall@{TOP_K}={recall_sum/n:.0%}  "
          f"precision@{TOP_K}={prec_sum/n:.2f}*  MRR@{TOP_K}={rr_sum/n:.3f}  (n={n})\n")
    return {"n": n, "hits": hits, "recall": recall_sum, "prec": prec_sum, "rr": rr_sum}


def evaluate(retriever=search_source_paths, retriever_name: str = "dense (search_source_paths)"):
    """전체 eval 실행. 하이브리드 검색기를 만들면 evaluate(hybrid_retriever, "hybrid")로
    호출해 dense와 같은 문항으로 비교하면 된다."""
    print(f"### retriever = {retriever_name}\n")
    print("=== primary (header-worded) ===")
    p = _run("primary", EVAL_SET, retriever)
    print("=== hard (paraphrased, avoids header vocab) ===")
    h = _run("hard", HARD_EVAL_SET, retriever)
    print("=== korean (TIL/writer corpus, paraphrased) — 아래 COMBINED엔 미포함, 별도 지표 ===")
    _run("korean", KOREAN_EVAL_SET, retriever)

    n = p["n"] + h["n"]
    print(f"=== COMBINED ({retriever_name}) ===")
    print(f"hit-rate@{TOP_K}={(p['hits']+h['hits'])/n:.0%}  "
          f"recall@{TOP_K}={(p['recall']+h['recall'])/n:.0%}  "
          f"precision@{TOP_K}={(p['prec']+h['prec'])/n:.2f}*  "
          f"MRR@{TOP_K}={(p['rr']+h['rr'])/n:.3f}  (n={n})")
    print("* precision@k: 문항당 정답을 1개만 라벨해 이론상 상한이 1/k(≈0.2)라 검색 품질보다\n"
          "  라벨 희소성을 반영한다. 순위 품질은 MRR로 본다(문서 docstring 참고).")


def _top1_score(vectorstore, question: str) -> tuple[float, str | None]:
    """top-1 문서의 relevance score와 source path. 결과 없으면 (0.0, None)."""
    results = vectorstore.similarity_search_with_relevance_scores(question, k=1)
    if not results:
        return 0.0, None
    doc, score = results[0]
    return score, doc.metadata.get("source")


def evaluate_threshold():
    """RELEVANCE_THRESHOLD(grade_docs가 "이 정도면 답해도 되나" 가르는 값) 캘리브레이션.
    grade_docs와 같은 similarity_search_with_relevance_scores로 top-1 점수를 모아, "top-1이
    실제로 정답 문서였던 경우"(양성)와 "애초에 코퍼스에 답이 없어야 하는 경우"(음성)의 점수
    분포를 비교한다. `uv run python scripts/eval_retrieval.py --threshold`로 실행 — 임베딩이
    로컬 모델(bge-m3)이라 API 비용은 없지만 첫 실행은 모델 로드로 느릴 수 있다."""
    vectorstore = get_vectorstore(build_embeddings())

    positive_scores = []
    for question, expected in EVAL_SET + HARD_EVAL_SET + KOREAN_EVAL_SET:
        score, source = _top1_score(vectorstore, question)
        if source in expected:  # top-1이 실제로 정답을 맞춘 경우만 — 틀린 랭킹은 별개 문제(MRR)
            positive_scores.append(score)

    negative_scores = [
        _top1_score(vectorstore, q)[0] for q in NEGATIVE_EVAL_SET + HARD_NEGATIVE_EVAL_SET
    ]

    def _stats(label, scores):
        if not scores:
            print(f"{label}: (no data)")
            return
        spread = f"  stdev={statistics.stdev(scores):.3f}" if len(scores) > 1 else ""
        print(f"{label}: n={len(scores)}  mean={statistics.mean(scores):.3f}  "
              f"median={statistics.median(scores):.3f}  min={min(scores):.3f}  max={max(scores):.3f}{spread}")

    print("=== top-1 relevance score distribution ===")
    _stats("positive (top-1 correctly matched gold doc)", positive_scores)
    _stats("negative (off-topic / out-of-corpus queries)", negative_scores)
    print(f"\ncurrent RELEVANCE_THRESHOLD = {RELEVANCE_THRESHOLD}\n")

    print("=== threshold sweep (recall = % positives kept, false-accept = % negatives wrongly kept) ===")
    print(f"{'threshold':>9}  {'recall':>7}  {'false-accept':>13}")
    for i in range(1, 19):
        t = i * 0.05
        recall = sum(s > t for s in positive_scores) / len(positive_scores) if positive_scores else 0
        false_accept = sum(s > t for s in negative_scores) / len(negative_scores) if negative_scores else 0
        marker = "  <- current" if abs(t - RELEVANCE_THRESHOLD) < 0.025 else ""
        print(f"{t:9.2f}  {recall:6.0%}  {false_accept:12.0%}{marker}")

    print("\n기준: recall은 최대한 유지하면서 false-accept이 눈에 띄게 꺾이는 지점을 고른다.")
    print("두 분포가 크게 겹치면 top_score 단일 신호로는 못 가른다는 뜻 — 하이브리드 붙여도")
    print("재조정이 또 필요할 가능성이 높다는 신호로 봐야 한다.")


_bm25_cache: dict = {}


def hybrid_search_source_paths(question, k=TOP_K):
    """search_source_paths와 같은 시그니처의 dense+BM25(RRF) 검색기. BM25 인덱스는 실제
    서빙 컬렉션과 같은 코퍼스(load_split_docs())로 한 번만 만들어 재사용한다."""
    if "index" not in _bm25_cache:
        docs = load_split_docs()
        _bm25_cache["docs"] = docs
        _bm25_cache["index"] = build_bm25(docs)
    vectorstore = get_vectorstore(build_embeddings())
    dense_docs = vectorstore.similarity_search(question, k=k)
    bm25_docs = bm25_search(_bm25_cache["index"], _bm25_cache["docs"], question, k=k)
    return _extract_sources(hybrid_merge(dense_docs, bm25_docs, k=k))


def compare_models():
    """모델별로 같은 문항을 돌려 검색 품질을 비교한다.
    주의: EVAL_SET/HARD_EVAL_SET의 정답 문서가 전부 영어(cs231n)라 이 비교는 코퍼스의
    영어 절반만 잰다. 한국어 문서(TIL/writer) 검색 품질은 별도 문항을 만들어야 잴 수 있다."""
    for label, model_name, collection in COMPARE_MODELS:
        build_embeddings.cache_clear()  # maxsize=1이라 모델 교체 시 이전 모델 해제
        print(f"\n{'=' * 70}\n### {label} — {model_name}\n{'=' * 70}")
        vectorstore = get_vectorstore(build_embeddings(model_name), collection)

        def retrieve(question, k=TOP_K, _vs=vectorstore):
            return _extract_sources(_vs.similarity_search(question, k=k))

        evaluate(retrieve, label)


if __name__ == "__main__":
    if "--threshold" in sys.argv:
        evaluate_threshold()
    elif "--compare" in sys.argv:
        compare_models()
    elif "--hybrid" in sys.argv:
        evaluate(hybrid_search_source_paths, "hybrid (dense+BM25 RRF)")
    else:
        evaluate()
