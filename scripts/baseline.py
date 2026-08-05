import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client
from langsmith.evaluation import evaluate

from app.rag.chain import TOP_K
from app.rag.graph import build_rag_graph

# 서빙과 동일한 그래프 (인덱싱은 build_rag_graph 내부에서 1회 수행)
rag = build_rag_graph()

_demo = rag.invoke({"question": "What is the main challenge of image classification?"})
print(_demo["answer"])
print("출처:", _demo["sources"])

# ---------------- 평가 ----------------
DATASET_NAME = "cs231n-rag-eval"
client = Client()

EVAL_QUESTIONS = [
    {
        "question": "What is the main challenge of image classification compared to other classification tasks?",
        "answer":   "Image classification faces the semantic gap and viewpoint variation, illumination changes, deformation, occlusion, background clutter, and intra-class variation, making it hard to write explicit rules for recognizing objects.",
    },
    {
        "question": "How does backpropagation compute gradients in a neural network?",
        "answer":   "Backpropagation applies the chain rule recursively from the loss output back through each layer, computing the gradient of the loss with respect to each parameter by multiplying local gradients along the computational graph.",
    },
    {
        "question": "What is the role of batch normalization in training deep neural networks?",
        "answer":   "Batch normalization normalizes activations within each mini-batch to have zero mean and unit variance, then applies learnable scale and shift parameters. This reduces internal covariate shift, allows higher learning rates, and acts as a regularizer.",
    },
    {
        "question": "What architectural innovation allowed ResNet to train very deep networks?",
        "answer":   "ResNet introduced residual (skip) connections that let gradients flow directly through identity shortcuts, making it possible to train networks with hundreds of layers without vanishing gradient problems.",
    },
    {
        "question": "How does a convolutional layer differ from a fully connected layer?",
        "answer":   "A convolutional layer applies shared filter weights locally across the spatial dimensions of the input, preserving spatial structure and drastically reducing parameters, while a fully connected layer connects every input neuron to every output neuron with independent weights.",
    },
    {
        "question": "What problem with long sequences does an LSTM address compared to a vanilla RNN?",
        "answer":   "Vanilla RNNs struggle to carry information across many time steps because gradients vanish or explode during backpropagation through time. LSTMs add a cell state with gated (input/forget/output) updates that let information flow largely unchanged across steps, making long-range dependencies learnable.",
    },
    {
        "question": "What is an adversarial example in the context of image classifiers?",
        "answer":   "An adversarial example is an input crafted by adding a small, often human-imperceptible perturbation to a correctly classified image so that the model misclassifies it with high confidence, revealing that the model's decision boundary doesn't align with human perception.",
    },
    {
        "question": "When is transfer learning particularly useful?",
        "answer":   "Transfer learning is useful when you have a small dataset for your target task but can start from a model pretrained on a large, related dataset — reusing its learned features avoids training from scratch and generally improves performance with less data.",
    },
    {
        # data/processed/week-05-TIL.md, Day 4 — Transformer masking 섹션 원문 기반.
        "question": "이 프로젝트의 학습 노트에 따르면, Transformer의 masking 기법은 무엇을 하는 기술이야?",
        "answer":   "Masking은 Attention이 보면 안 되는 위치를 가리는 기법이다.",
    },
    {
        # data/processed/week-08-TIL.md 요약 원문 기반.
        "question": "학습 노트에 따르면 LangGraph의 Checkpointer는 어떤 것을 가능하게 해줘?",
        "answer":   "Checkpointer는 그래프 실행을 멈췄다가 이어갈 수 있게 해주는 영속성(durable execution)을 확보해준다.",
    },
]
print(f"검증 질문 수: {len(EVAL_QUESTIONS)}")

# Dataset 생성 or 재사용. 재사용일 땐 기존 예제엔 없는 질문만 추가한다 — 처음엔 EVAL_QUESTIONS를
# 한 번만 밀어넣고 끝이라 재실행해도 새로 추가한 문항이 조용히 누락됐었다(TIL 문항 2개 추가하며 발견).
existing = [d for d in client.list_datasets(dataset_name=DATASET_NAME)]
if existing:
    dataset = existing[0]
    known_questions = {
        ex.inputs.get("question") for ex in client.list_examples(dataset_id=dataset.id)
    }
    new_qs = [ex for ex in EVAL_QUESTIONS if ex["question"] not in known_questions]
    if new_qs:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"question": ex["question"]} for ex in new_qs],
            outputs=[{"answer": ex["answer"]} for ex in new_qs],
        )
    print(f"기존 Dataset 사용: {dataset.id} (신규 {len(new_qs)}건 추가)")
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="CS231n 강의 노트 기반 RAG 답변 품질 평가용",
    )
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question": ex["question"]} for ex in EVAL_QUESTIONS],
        outputs=[{"answer": ex["answer"]} for ex in EVAL_QUESTIONS],
    )
    print(f"새 Dataset 생성: {dataset.id} (Example {len(EVAL_QUESTIONS)}건)")


def target(inputs):
    result = rag.invoke({"question": inputs["question"]})
    return {"answer": result["answer"], "sources": result["sources"]}


# 휴리스틱: 기대 답변의 핵심 내용어(영문 4자↑ 또는 한글 2자↑, 불용어 제외) 회수율(0~1).
# 한글 토큰(가-힣 2자↑)을 추가한 이유: TIL 노트 기반 한글 평가 문항을 넣으면서, 원래
# 영문 전용 정규식이 한글 답변에서 키워드를 하나도 못 뽑아 항상 score=0으로 떨어졌음.
_STOPWORDS = {
    "that", "this", "with", "from", "into", "each", "then", "than", "when",
    "what", "which", "while", "these", "those", "their", "them", "they",
    "other", "such", "compared", "using", "make", "makes", "made", "does",
    "where", "have", "been", "also", "between", "across", "along", "every",
}


def keyword_recall(run, example):
    pred = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    keywords = {w for w in re.findall(r"[a-z]{4,}|[가-힣]{2,}", expected) if w not in _STOPWORDS}
    if not keywords:
        return {"key": "keyword_recall", "score": 0, "comment": "기대 답변에 키워드 없음"}
    hit = {w for w in keywords if w in pred}
    return {
        "key": "keyword_recall",
        "score": round(len(hit) / len(keywords), 3),
        "comment": f"기대 키워드 {len(keywords)}개 중 {len(hit)}개 포함",
    }


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a grader assessing answer quality. "
     "Compare the reference answer with the model's prediction and assign a score: "
     "1 if they match in meaning, 0.5 if only partially, 0 if unrelated. "
     "Output ONLY the number (0, 0.5, or 1) on the first line, "
     "then a brief reason on the following lines."),
    ("human",
     "Question: {question}\n\n"
     "Reference answer: {reference}\n\n"
     "Model answer: {prediction}"),
])

judge_llm = ChatGoogleGenerativeAI(
    # 판정 모델은 생성 모델(GOOGLE_MODEL)과 분리 — self-bias 최소화(1sanguk 교훈).
    model=os.getenv("GOOGLE_EVAL_MODEL", "gemini-3.5-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)
RELEVANCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Does the answer directly address the question? "
     "Score 1 (fully addresses), 0.5 (partially), 0 (does not address). "
     "Output ONLY the number (0, 0.5, or 1) on the first line, "
     "then a brief reason on the following lines."),
    ("human",
     "Question: {question}\n\n"
     "Answer: {answer}"),
])

judge_chain = JUDGE_PROMPT | judge_llm | StrOutputParser()
relevancy_chain = RELEVANCY_PROMPT | judge_llm | StrOutputParser()


def llm_judge(run, example):
    # 호출 실패(429 등) 시 score=None으로 평가 전체가 죽지 않게 감싼다.
    try:
        reply = judge_chain.invoke({
            "question": example.inputs["question"],
            "reference": example.outputs["answer"],
            "prediction": run.outputs["answer"],
        })
    except Exception as e:
        return {"key": "llm_judge_semantic_match", "score": None, "comment": f"judge 호출 실패: {e}"}
    first_line = reply.strip().splitlines()[0].strip()
    try:
        score = float(first_line)
    except ValueError:
        # 파싱 실패는 "0점(완전 오답)"이 아니라 "판정 불가"다 — 0으로 떨어뜨리면 평균이
        # 거짓으로 낮아진다(SWHee/SungjinWi99도 이 구분을 지킨다). None으로 두고 원문을 남긴다.
        return {"key": "llm_judge_semantic_match", "score": None,
                "comment": f"judge 응답 파싱 실패(첫 줄이 숫자가 아님): {reply!r}"}
    return {"key": "llm_judge_semantic_match", "score": score, "comment": reply}


def answer_relevancy(run, example):
    try:
        reply = relevancy_chain.invoke({
            "question": example.inputs["question"],
            "answer": run.outputs.get("answer", ""),
        })
    except Exception as e:
        return {"key": "answer_relevancy", "score": None, "comment": f"judge 호출 실패: {e}"}
    first_line = reply.strip().splitlines()[0].strip()
    try:
        score = float(first_line)
    except ValueError:
        # llm_judge와 같은 이유로 0이 아니라 None — 파싱 실패는 "판정 불가"지 "오답"이 아니다.
        return {"key": "answer_relevancy", "score": None,
                "comment": f"judge 응답 파싱 실패(첫 줄이 숫자가 아님): {reply!r}"}
    return {"key": "answer_relevancy", "score": score, "comment": reply}


# 실험 라벨: 임베딩 모델은 고정이라 변별력이 없으므로 '생성 LLM'으로 구분한다.
_provider = os.getenv("LLM_PROVIDER", "google").lower()
_llm_name = (os.getenv("OLLAMA_MODEL", "gemma4:e2b") if _provider == "ollama"
             else os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"))
_llm_label = _llm_name.replace(":", "-").replace("/", "-")

result = evaluate(
    target,
    data=DATASET_NAME,
    evaluators=[keyword_recall, llm_judge, answer_relevancy],
    experiment_prefix=f"{_llm_label}-k{TOP_K}-{datetime.now().strftime('%m%d%H%M')}",
)
print(result)
