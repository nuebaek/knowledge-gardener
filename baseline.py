from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import chromadb
import re
import time
import os
from pathlib import Path


class RateLimitedEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts):
        batch_size = 90
        results = []
        for i in range(0, len(texts), batch_size):
            if i > 0:
                time.sleep(1.0)
            batch = texts[i:i + batch_size]
            print(f"  임베딩 중... {min(i + batch_size, len(texts))}/{len(texts)}")
            results.extend(super().embed_documents(batch))
        return results

load_dotenv()

# 인덱싱
print("문서 로딩 및 인덱싱 시작...")

# 2단계 청킹: 마크다운 헤더로 섹션 분할(의미 단위 보존) 후 긴 섹션만 길이 분할.
DATA_DIR = Path(__file__).parent / "data" / "processed"

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
    strip_headers=False,  # 헤더 텍스트를 청크에 남겨 임베딩 신호로 활용
)
char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

md_files = sorted(DATA_DIR.glob("**/*.md"))
print(f"로딩된 문서 수: {len(md_files)}")

split_docs = []
for path in md_files:
    sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
    for sec in sections:
        sec.metadata["source"] = str(path)
    split_docs.extend(char_splitter.split_documents(sections))
print(f"분할된 chunk 수: {len(split_docs)}")

# 임베딩 provider(EMBED_PROVIDER): local=bge-base / google=gemini-embedding-001.
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").lower()
if EMBED_PROVIDER == "google":
    print("임베딩: Gemini gemini-embedding-001")
    embeddings = RateLimitedEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
else:
    print("임베딩: 로컬 bge-base-en-v1.5")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# Chroma 연결(CHROMA_MODE): local=임베디드 PersistentClient(기본) / server=별도 서버 / cloud=Cloud(무료 300개 제한).
CHROMA_COLLECTION = "cs231n"
CHROMA_MODE = os.getenv("CHROMA_MODE", "local").lower()
if CHROMA_MODE == "cloud":
    print(f"Chroma Cloud 연결 (tenant={os.getenv('CHROMA_TENANT')}, db={os.getenv('CHROMA_DATABASE')})")
    chroma_client = chromadb.CloudClient(
        tenant=os.getenv("CHROMA_TENANT"),
        database=os.getenv("CHROMA_DATABASE"),
        api_key=os.getenv("CHROMA_API_KEY"),
    )
elif CHROMA_MODE == "server":
    print("Chroma 서버 연결 (localhost)")
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
    )
else:
    persist_path = str(Path(__file__).parent / "chroma_data")
    print(f"로컬 Chroma(임베디드) 연결: {persist_path}")
    chroma_client = chromadb.PersistentClient(path=persist_path)


existing = {c.name for c in chroma_client.list_collections()}
need_index = True
if CHROMA_COLLECTION in existing:
    count = chroma_client.get_collection(CHROMA_COLLECTION).count()
    if count > 0:
        print(f"기존 컬렉션 '{CHROMA_COLLECTION}' 재사용 ({count}개 임베딩, 생략)")
        vectorstore = Chroma(
            client=chroma_client,
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
        )
        need_index = False
    else:
        print(f"빈 컬렉션 '{CHROMA_COLLECTION}' 발견 → 삭제 후 재인덱싱")
        chroma_client.delete_collection(CHROMA_COLLECTION)

if need_index:
    print(f"새 컬렉션 '{CHROMA_COLLECTION}' 생성 및 임베딩 중...")
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        client=chroma_client,
        collection_name=CHROMA_COLLECTION,
    )

print("인덱싱 완료")

# RAG
print("RAG 파이프라인 시작...")
TOP_K = 5  # 검색 청크 수
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# 답변 생성 프롬프트 (답변 언어는 질문 언어를 따른다)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's question using ONLY the provided documents. "
     "Always reply in the SAME language as the question "
     "(a Korean question must be answered in Korean, an English question in English). "
     "If the documents are insufficient, say so honestly instead of guessing.\n\n"
     "{context}"),
    ("human", "{question}"),
])

def build_llm():
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    print(f"LLM Provider: {provider}")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    return ChatGoogleGenerativeAI(
        model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


llm = build_llm()

def format_docs(ds):
    return "\n\n".join(d.page_content for d in ds)

def extract_sources(ds):
    """검색된 청크의 출처(파일 경로)를 중복 제거해 리스트로 반환."""
    sources = []
    for d in ds:
        src = d.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)
    return sources

# 1) 질문 → 검색. 검색을 한 번만 수행해 답변 생성과 출처 추출이 같은 docs를 공유한다.
retrieve = RunnableParallel(
    docs=retriever,
    question=RunnablePassthrough(),
)

# 2) {docs, question} → 답변 문자열
generate = (
    {
        "context": lambda x: format_docs(x["docs"]),
        "question": lambda x: x["question"],
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 3) 최종: 답변 + 출처를 함께 반환 (과제 요구사항: 응답에 출처 포함)
rag = retrieve | RunnableParallel(
    answer=generate,
    sources=lambda x: extract_sources(x["docs"]),
)

_demo = rag.invoke("What is the main challenge of image classification?")
print(_demo["answer"])
print("출처:", _demo["sources"])

print("RAG 파이프라인 완료")

# 평가
from langsmith.evaluation import evaluate
from langsmith import Client

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

existing = [d for d in client.list_datasets(dataset_name=DATASET_NAME)]

inputs  = [{"question": ex["question"]} for ex in EVAL_QUESTIONS]
outputs = [{"answer":   ex["answer"]}   for ex in EVAL_QUESTIONS]

if existing:
    dataset = existing[0]
    print(f"기존 Dataset 사용: {dataset.id}")
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="CS231n 강의 노트 기반 RAG 답변 품질 평가용",
    )
    print(f"새 Dataset 생성: {dataset.id}")
    client.create_examples(
        dataset_id=dataset.id,
        inputs=inputs,
        outputs=outputs,
    )
    print(f"Example {len(EVAL_QUESTIONS)}건 추가 완료")

loaded = client.read_dataset(dataset_name=DATASET_NAME)

examples = list(client.list_examples(dataset_id=loaded.id))
print(f"총 Example 수: {len(examples)}")

for ex in examples[:3]:
    print("Q:", ex.inputs["question"])
    print("A:", ex.outputs["answer"] if ex.outputs else "(없음)")
    print()

def target(inputs):
    result = rag.invoke(inputs["question"])
    return {"answer": result["answer"], "sources": result["sources"]}

# 휴리스틱 평가기: 기대 답변의 핵심 내용어(4자↑, 불용어 제외) 회수율(0~1).
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
    # 4글자 이상 영문 단어 중 불용어 제외 → 핵심 내용어 집합
    keywords = {w for w in re.findall(r"[a-z]{4,}", expected) if w not in _STOPWORDS}
    if not keywords:
        return {"key": "keyword_recall", "score": 0, "comment": "기대 답변에 키워드 없음"}
    hit = {w for w in keywords if w in pred}
    score = len(hit) / len(keywords)
    return {
        "key": "keyword_recall",
        "score": round(score, 3),
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

judge_chain = JUDGE_PROMPT | judge_llm | StrOutputParser()

def llm_judge(run, example):
    # Judge 호출이 실패해도(429 등) 전체 평가가 죽지 않게 감싼다.
    # 실패 시 score=None → LangSmith가 해당 점수를 건너뛰고, keyword_recall은 그대로 남는다.
    try:
        reply = judge_chain.invoke({
            "question": example.inputs["question"],
            "reference": example.outputs["answer"],
            "prediction": run.outputs["answer"],
        })
    except Exception as e:
        return {
            "key": "llm_judge_semantic_match",
            "score": None,
            "comment": f"judge 호출 실패: {e}",
        }
    # === 첫 줄의 숫자만 점수로 사용 ===
    first_line = reply.strip().splitlines()[0].strip()
    try:
        score = float(first_line)
    except ValueError:
        score = 0
    return {
        "key": "llm_judge_semantic_match",
        "score": score,
        "comment": reply,
    }

result = evaluate(
    target,
    data=DATASET_NAME,
    evaluators=[keyword_recall, llm_judge],
    experiment_prefix=f"{EMBED_PROVIDER}-k{TOP_K}",  # 설정별로 라벨 자동 구분
)

print(result)