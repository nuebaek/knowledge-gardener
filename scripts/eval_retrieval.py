"""Retrieval eval — LLM judge 없이 결정적으로 검색 품질을 잰다. `uv run python
scripts/eval_retrieval.py`로 실행. 지표 3개(전부 정답 문서 라벨만으로 계산):

  - hit-rate@k : top-k에 정답이 하나라도 들어오나 (=정답 1개면 recall과 동일)
  - recall@k   : 정답 중 몇 개를 회수했나 (정답 여러 개인 문항에서 hit-rate와 갈림)
  - MRR@k      : 정답이 몇 등에 나오나 (순위 품질). hit-rate가 이미 포화(92%)라
                 하이브리드 검색의 개선은 hit-rate가 아니라 MRR에서 드러난다.

  - precision@k: top-k 중 정답 비율. 문항당 정답을 1개만 라벨해 이론상 상한이 1/k(≈0.2)
                 라서 검색 품질보다 라벨 희소성을 반영한다(*로 표기). 제대로 재려면 SWHee처럼
                 문항마다 관련 문서를 전부(필수+보조) 라벨해야 함 — 참고용으로만 본다.

evaluate(retriever)로 검색기를 갈아끼우면 같은 문항으로 dense vs hybrid를 비교할 수 있다.
week-*-TIL.md는 회고문이라 "정답 문서" 개념이 안 맞아 제외. 개념이 여러 파일에 걸친 문항은
expected를 리스트로 둬 그중 하나만 맞아도 hit으로 센다(코퍼스 품질 이슈라 결과에 같이 남음).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chain import TOP_K, search_source_paths  # noqa: E402

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


def _reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    """정답이 처음 나오는 순위의 역수. top-k 안에 없으면 0. MRR의 문항별 값."""
    for i, path in enumerate(retrieved, start=1):
        if path in gold:
            return 1.0 / i
    return 0.0


def _run(label: str, eval_set: list, retriever) -> dict:
    """검색기(retriever: query,k -> [path])를 주입받아 문항별로 3지표 계산.
    retriever를 갈아끼우면 같은 eval로 dense vs hybrid를 비교할 수 있다(1sanguk 패턴).
    hit-rate: top-k에 정답이 하나라도 있나(any) / recall: 정답 중 몇 개를 회수했나 /
    MRR: 정답이 몇 등에 나오나(순위 품질 — 하이브리드 효과가 여기서 드러난다)."""
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

    n = p["n"] + h["n"]
    print(f"=== COMBINED ({retriever_name}) ===")
    print(f"hit-rate@{TOP_K}={(p['hits']+h['hits'])/n:.0%}  "
          f"recall@{TOP_K}={(p['recall']+h['recall'])/n:.0%}  "
          f"precision@{TOP_K}={(p['prec']+h['prec'])/n:.2f}*  "
          f"MRR@{TOP_K}={(p['rr']+h['rr'])/n:.3f}  (n={n})")
    print("* precision@k: 문항당 정답을 1개만 라벨해 이론상 상한이 1/k(≈0.2)라 검색 품질보다\n"
          "  라벨 희소성을 반영한다. 순위 품질은 MRR로 본다(문서 docstring 참고).")


if __name__ == "__main__":
    evaluate()
