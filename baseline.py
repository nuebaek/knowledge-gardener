import os
import re
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client
from langsmith.evaluation import evaluate

from rag import TOP_K, build_rag_chain

# 서빙과 동일한 체인 (인덱싱은 build_rag_chain 내부에서 1회 수행)
rag = build_rag_chain()

_demo = rag.invoke("What is the main challenge of image classification?")
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
]
print(f"검증 질문 수: {len(EVAL_QUESTIONS)}")

# Dataset 생성 or 재사용
existing = [d for d in client.list_datasets(dataset_name=DATASET_NAME)]
if existing:
    dataset = existing[0]
    print(f"기존 Dataset 사용: {dataset.id}")
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
    result = rag.invoke(inputs["question"])
    return {"answer": result["answer"], "sources": result["sources"]}


# 휴리스틱: 기대 답변의 핵심 내용어(4자↑, 불용어 제외) 회수율(0~1).
# (영문 토큰 기준 — 한글 평가 문항을 추가하면 토큰화 보강 필요)
_STOPWORDS = {
    "that", "this", "with", "from", "into", "each", "then", "than", "when",
    "what", "which", "while", "these", "those", "their", "them", "they",
    "other", "such", "compared", "using", "make", "makes", "made", "does",
    "where", "have", "been", "also", "between", "across", "along", "every",
}


def keyword_recall(run, example):
    pred = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    keywords = {w for w in re.findall(r"[a-z]{4,}", expected) if w not in _STOPWORDS}
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
    model=os.getenv("JUDGE_MODEL", "gemini-3.5-flash"),
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
        score = 0
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
        score = 0
    return {"key": "answer_relevancy", "score": score, "comment": reply}


# 실험 라벨: 임베딩(bge-m3)은 고정이라 변별력이 없으므로 '생성 LLM'으로 구분한다.
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
