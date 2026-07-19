import os

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

from app.core import catalog
from app.core.paths import CHROMA_DIR, PROJECT_ROOT

load_dotenv()

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
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _make_splitters():
    # 1단계: 헤더 기준 분할로 섹션 단위 의미 보존. 2단계: 긴 섹션만 길이 기준으로 추가 분할.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return header_splitter, char_splitter


def _split_path(path, header_splitter, char_splitter) -> list:
    # source를 절대경로가 아니라 catalog의 source_path(PROJECT_ROOT 기준 상대경로)로 맞춰야
    # sync_index()의 delete(where={"source": ...})가 같은 문서의 기존 청크를 정확히 찾는다.
    source = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
    for sec in sections:
        sec.metadata["source"] = source
    chunks = char_splitter.split_documents(sections)
    for chunk in chunks:
        inject_header_context(chunk)
    return chunks


def load_split_docs(rows=None):
    """rows를 안 주면 카탈로그 전체(초기 구축용), 주면 그 문서들만(증분용) 청킹한다.

    예전엔 data/processed만 훑었어서 writer가 쓴 daily/weekly/til은 답변 생성에서 아예
    검색이 안 됐다 — catalog가 단일 소스가 된 지금은 doc_type을 안 가리고 다 포함시킨다.
    """
    header_splitter, char_splitter = _make_splitters()
    if rows is None:
        rows = catalog.list_documents()
    split_docs = []
    for row in rows:
        split_docs.extend(_split_path(PROJECT_ROOT / row["source_path"], header_splitter, char_splitter))
    return split_docs


def inject_header_context(doc) -> None:
    # page_content에 헤더 정보를 metadata에서 복원
    m = doc.metadata
    parts = [m.get("h1"), m.get("h2"), m.get("h3")]
    prefix = " > ".join(p for p in parts if p)
    if prefix and not doc.page_content.lstrip().startswith("#"):
        doc.page_content = f"{prefix}\n\n{doc.page_content}"


def sync_index(vectorstore) -> int:
    """catalog에서 indexed_at IS NULL인(새로 생기거나 내용이 바뀐) 문서만 재청킹해서
    그 문서의 기존 청크만 지우고 새로 넣는다 — 컬렉션 전체를 다시 만들 필요가 없다."""
    header_splitter, char_splitter = _make_splitters()
    pending = catalog.pending_reindex()
    for row in pending:
        path = PROJECT_ROOT / row["source_path"]
        if not path.exists():
            continue  # 파일이 지워진 케이스는 스코프 아웃
        vectorstore.delete(where={"source": row["source_path"]})
        chunks = _split_path(path, header_splitter, char_splitter)
        if chunks:
            vectorstore.add_documents(chunks)
        catalog.mark_indexed(row["source_path"])
    return len(pending)


def get_vectorstore(embeddings):
    client = build_chroma_client()
    existing = {c.name for c in client.list_collections()}
    if CHROMA_COLLECTION in existing and client.get_collection(CHROMA_COLLECTION).count() > 0:
        vectorstore = Chroma(client=client, collection_name=CHROMA_COLLECTION, embedding_function=embeddings)
        sync_index(vectorstore)
        return vectorstore

    if CHROMA_COLLECTION in existing:
        client.delete_collection(CHROMA_COLLECTION)

    rows = catalog.list_documents()
    vectorstore = Chroma.from_documents(
        load_split_docs(rows), embeddings, client=client, collection_name=CHROMA_COLLECTION,
    )
    for row in rows:
        catalog.mark_indexed(row["source_path"])
    return vectorstore


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


AGENT_SYSTEM_PROMPT = (
    "You are a study-assistant agent. You have four tools: `answer_question`, `write_daily`, "
    "`write_weekly`, and `write_til`. Each tool's own description explains exactly when to use "
    "it and what fields it needs — rely on those, not on this summary, to decide.\n\n"
    "MUST: whenever the user asks a question, requests an explanation, or asks what "
    "something is or how it works, call `answer_question`. Never answer such questions "
    "from your own knowledge, even if you are confident you know the answer.\n\n"
    "MUST: whenever the user wants to record a retrospective, call `write_til` or `write_weekly` "
    "as appropriate. Never invent field values the user did not state — if a required field is "
    "missing, ask the user for it instead of guessing.\n\n"
    "When you detect that the user wants to record today's study session (회고형 발화), do NOT "
    "call `write_daily` immediately. Follow this retrieval-first flow:\n\n"
    "1. From what the user said (or from any study material / memo they pasted), list the "
    "distinct keywords or topics covered today and show the list. Do not explain any of them.\n"
    "2. For each topic, ask the user to explain it in their own words (\"이건 네 말로 설명하면 "
    "뭐야?\"). Ask at most 1 follow-up question per topic by default, 2 at the absolute maximum "
    "(e.g. \"왜 그렇게 동작해?\", \"언제 쓰는 거야?\"). Never exceed 2.\n"
    "3. If the user says they don't know, or says to skip (\"모르겠어\", \"넘어가\", \"이건 됐어\"), "
    "move on immediately and remember that this topic was NOT explained — it must appear in the "
    "note as a `🌱 다시 꺼내볼 것` item. Never explain it yourself and never fill it from the "
    "material.\n"
    "4. If the user pasted study material, you may use it only to (a) build the topic list and "
    "(b) point out a mismatch when the user's explanation contradicts it (\"자료엔 다르게 돼 있는데 "
    "다시 볼래?\"). Never read the material's content out as the explanation.\n"
    "5. When all topics are done, or when the user asks to save now, call `write_daily` with "
    "`learned` set to the user's own explanations concatenated verbatim (including notes on what "
    "they could not explain), `topic` and `related_concepts` taken only from what was actually "
    "discussed.\n"
    "6. Low-friction fallback: if the user signals they want to skip the conversation entirely "
    "(\"오늘은 그냥 저장만 해줘\", \"인출 없이 저장\"), respect it without pushback — call "
    "`write_daily` with their raw memo as `learned`. A blurry note saved today beats a perfect "
    "note never written.\n\n"
    "If the user only greets you or makes small talk, respond directly without calling any tool."
)


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
