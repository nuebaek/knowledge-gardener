import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_cerebras import ChatCerebras
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
CHROMA_COLLECTION = "cs231n"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.4


# ---------------- 임베딩 & 벡터스토어 ----------------

def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_chroma_client():
    mode = os.getenv("CHROMA_MODE", "local").lower()
    if mode == "cloud":
        return chromadb.CloudClient(
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
            api_key=os.getenv("CHROMA_API_KEY"),
        )
    if mode == "server":
        return chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
    return chromadb.PersistentClient(path=str(BASE_DIR / "chroma_data"))


def load_split_docs():
    # 1단계: 헤더 기준 분할로 섹션 단위 의미 보존
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    # 2단계: 긴 섹션만 길이 기준으로 추가 분할
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = []
    for path in sorted(DATA_DIR.glob("**/*.md")):
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        for sec in sections:
            sec.metadata["source"] = str(path)
        chunks = char_splitter.split_documents(sections)
        for chunk in chunks:
            inject_header_context(chunk)
        split_docs.extend(chunks)
    return split_docs


def inject_header_context(doc) -> None:
    # page_content에 헤더 정보를 metadata에서 복원
    m = doc.metadata
    parts = [m.get("h1"), m.get("h2"), m.get("h3")]
    prefix = " > ".join(p for p in parts if p)
    if prefix and not doc.page_content.lstrip().startswith("#"):
        doc.page_content = f"{prefix}\n\n{doc.page_content}"


def get_vectorstore(embeddings):
    client = build_chroma_client()
    existing = {c.name for c in client.list_collections()}
    if CHROMA_COLLECTION in existing:
        if client.get_collection(CHROMA_COLLECTION).count() > 0:
            return Chroma(
                client=client,
                collection_name=CHROMA_COLLECTION,
                embedding_function=embeddings,
            )
        client.delete_collection(CHROMA_COLLECTION)
    return Chroma.from_documents(
        load_split_docs(),
        embeddings,
        client=client,
        collection_name=CHROMA_COLLECTION,
    )


def get_retriever(embeddings, k: int = TOP_K):
    return get_vectorstore(embeddings).as_retriever(search_kwargs={"k": k})


# ---------------- LLM ----------------

def build_llm():
    # 생성 LLM만 provider를 고른다(LLM_PROVIDER).
    provider = os.getenv("LLM_PROVIDER", "cerebras").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    return ChatCerebras(
        model=os.getenv("CEREBRAS_MODEL", "gemma-4-31b"),
        api_key=os.getenv("CEREBRAS_API_KEY"),
    )


# ---------------- 프롬프트 & 체인 조립 ----------------

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document-grounded Q&A assistant. "
     "Answer using ONLY the provided documents — do not rely on prior knowledge or infer beyond what is explicitly written. "
     "Match your response language to the question (Korean → Korean, English → English) in EVERY case, "
     "including when you cannot answer. "
     "If the documents lack sufficient information, say so clearly in the question's own language — "
     "for example, Korean: \"제공된 자료에 이 내용이 없습니다.\" / English: \"The provided materials do not cover this.\" "
     "Never default to English when the question was asked in another language.\n\n"
     "{context}"),
    ("human", "{question}"),
])


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite search queries for a document retrieval system. "
     "The query below failed to retrieve relevant documents. "
     "Rewrite it using more precise or alternative technical terminology, "
     "or reframe it from a different angle, so that a new search against the same "
     "corpus is more likely to find relevant material. "
     "Keep the original intent and scope — do not broaden into a different topic, "
     "and do not answer the question yourself. "
     "Match the language of the original query exactly (Korean → Korean, English → English). "
     "Output ONLY the rewritten query as one line — no explanation, no prefix, no quotes."
    ),
    ("human",
     "Original question to rewrite: \"{question}\"\n"
     "A previous rewrite attempt also failed to retrieve relevant results: \"{rewritten_question}\"\n\n"
     "Write ONE new rewritten query, different from the previous attempt."),
])



def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def _extract_sources(docs):
    sources = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        if src not in sources:
            sources.append(src)
    return sources


def ingest():
    """인덱싱만 단독 실행. `uv run python rag.py`"""
    embeddings = build_embeddings()
    vectorstore = get_vectorstore(embeddings)
    print(f"인덱싱 완료: {vectorstore._collection.count()}개 청크")


def build_rag_chain():
    """인덱싱(필요 시) + LCEL 체인 구성. invoke(question) → {answer, sources}."""
    embeddings = build_embeddings()
    retriever = get_retriever(embeddings)
    llm = build_llm()

    # 검색을 1회만 수행해 답변 생성과 출처 추출이 같은 docs를 공유
    retrieve = RunnableParallel(docs=retriever, question=RunnablePassthrough())
    generate = (
        {
            "context": lambda x: _format_docs(x["docs"]),
            "question": lambda x: x["question"],
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return retrieve | RunnableParallel(
        answer=generate,
        sources=lambda x: _extract_sources(x["docs"]),
    )


if __name__ == "__main__":
    ingest()
