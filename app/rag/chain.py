import logging
import os
import re
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from rank_bm25 import BM25Okapi

from app.core import catalog
from app.core.paths import CHROMA_DIR, PROJECT_ROOT
from app.rag.prompts import (
    DOC_QA_PROMPT, RECALL_QUESTION_PROMPT, TURN_PROMPT, format_doc_chat_history,
)
from app.schemas.rag import TurnResult

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_COLLECTION = "cs231n_v2"
EMBED_MODEL = os.getenv("EMBED_MODEL", "dragonkue/multilingual-e5-small-ko-v2")
TOP_K = 5
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.47"))
RRF_K = int(os.getenv("RRF_K", "60"))


def study_turn(llm, topic, next_topic, message, conversation) -> TurnResult:
    turn = apply_fallback(llm, lambda m: m.with_structured_output(TurnResult))
    messages = TURN_PROMPT.invoke({
        "topic": topic,
        "next_topic": next_topic or "none",
        "message": message,
        "conversation": conversation,
    }).to_messages()
    return turn.invoke(messages)


@lru_cache(maxsize=1)
def build_embeddings(model_name: str | None = None):
    from langchain_huggingface import HuggingFaceEmbeddings

    name = model_name or EMBED_MODEL
    model_kwargs = {
        "device": "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu",
        "model_kwargs": {"use_safetensors": True},
    }
    encode_kwargs = {"normalize_embeddings": True}
    query_encode_kwargs = dict(encode_kwargs)
    # e5 계열은 query:/passage: 접두사 없이 쓰면 성능이 눈에 띄게 떨어진다.
    if "-e5-" in name:
        model_kwargs["prompts"] = {"query": "query: ", "passage": "passage: "}
        encode_kwargs["prompt_name"] = "passage"
        query_encode_kwargs["prompt_name"] = "query"
    return HuggingFaceEmbeddings(
        model_name=name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_encode_kwargs=query_encode_kwargs,
    )


def build_chroma_client():
    mode = os.getenv("CHROMA_MODE", "local").lower()
    if mode == "server":
        return chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", "8000")),
        )
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _make_splitters():
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return header_splitter, char_splitter


def _split_path(path, header_splitter, char_splitter) -> list:
    # source는 sync_index()의 delete(where={"source": ...})와 맞아야 하므로 상대경로 유지.
    source = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
    for sec in sections:
        sec.metadata["source"] = source
    chunks = char_splitter.split_documents(sections)
    for chunk in chunks:
        inject_header_context(chunk)
    return chunks


def load_split_docs(rows=None):
    header_splitter, char_splitter = _make_splitters()
    if rows is None:
        rows = catalog.list_documents()
    split_docs = []
    for row in rows:
        path = PROJECT_ROOT / row["source_path"]
        if not path.exists():
            continue
        split_docs.extend(_split_path(path, header_splitter, char_splitter))
    return split_docs


def section_path(meta) -> str:
    return " > ".join(p for p in (meta.get("h1"), meta.get("h2"), meta.get("h3")) if p)


def section_key(meta) -> str:
    return f"{meta.get('source', '?')}#{section_path(meta)}"


def inject_header_context(doc) -> None:
    prefix = section_path(doc.metadata)
    if prefix and not doc.page_content.lstrip().startswith("#"):
        doc.page_content = f"{prefix}\n\n{doc.page_content}"


def sync_index(vectorstore) -> int:
    header_splitter, char_splitter = _make_splitters()
    pending = catalog.pending_reindex()
    for row in pending:
        path = PROJECT_ROOT / row["source_path"]
        if not path.exists():
            continue
        vectorstore.delete(where={"source": row["source_path"]})
        chunks = _split_path(path, header_splitter, char_splitter)
        if chunks:
            vectorstore.add_documents(chunks)
        catalog.mark_indexed(row["source_path"])
    return len(pending)


_vectorstore_cache: dict = {}


def get_vectorstore(embeddings, collection: str = CHROMA_COLLECTION):
    if collection in _vectorstore_cache:
        return _vectorstore_cache[collection]
    vectorstore = _build_vectorstore(embeddings, collection)
    _vectorstore_cache[collection] = vectorstore
    return vectorstore


def _build_vectorstore(embeddings, collection: str) -> "Chroma":
    client = build_chroma_client()
    existing = {c.name for c in client.list_collections()}
    if collection in existing and client.get_collection(collection).count() > 0:
        vectorstore = Chroma(client=client, collection_name=collection, embedding_function=embeddings)
        sync_index(vectorstore)
        return vectorstore

    if collection in existing:
        client.delete_collection(collection)

    rows = catalog.list_documents()
    vectorstore = Chroma.from_documents(
        load_split_docs(rows), embeddings, client=client, collection_name=collection,
    )
    for row in rows:
        catalog.mark_indexed(row["source_path"])
    return vectorstore


_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")

# 체언/용언 어간만 남기고 조사(JX/JKO/...)·어미(EC/ETM/...)는 버린다 — "학습을"/"학습이"/
# "학습하는"이 전부 다른 토큰이 되던 문제(정규식은 조사·어미까지 통째로 묶어서 봄)의 원인.
_CONTENT_TAGS = {"NNG", "NNP", "NNB", "NP", "NR", "VV", "VA", "VX", "VCP", "VCN", "SL", "SH", "SN"}


@lru_cache(maxsize=1)
def _kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()


def _tokenize(text: str) -> list[str]:
    morphs = (t.form for t in _kiwi().tokenize(text) if t.tag in _CONTENT_TAGS)
    return [piece for form in morphs for piece in _TOKEN_RE.findall(form.lower())]


def build_bm25(docs: list) -> BM25Okapi:
    return BM25Okapi([_tokenize(d.page_content) for d in docs])


def bm25_search(bm25: BM25Okapi, docs: list, query: str, k: int) -> list:
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [docs[i] for i in ranked[:k] if scores[i] > 0]


def _doc_key(doc) -> str:
    return f"{doc.metadata.get('source', '?')}::{doc.page_content[:80]}"


def hybrid_merge(dense_docs: list, bm25_docs: list, k: int, rrf_k: int = 60) -> list:
    """Reciprocal Rank Fusion — 프로덕션에서는 안 씀(rerank_docs로 대체), eval 비교용으로만 유지."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, object] = {}
    for source in (dense_docs, bm25_docs):
        for rank, doc in enumerate(source, 1):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + rank)
            docs_by_key.setdefault(key, doc)
    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [docs_by_key[key] for key in ranked_keys[:k]]


RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_POOL_K = 15


@lru_cache(maxsize=1)
def build_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, max_length=512)


def rerank_docs(reranker, query: str, dense_docs: list, bm25_docs: list, k: int) -> tuple[list, list]:
    """dense+BM25 후보 풀을 cross-encoder로 reranking한다."""
    seen, pool = set(), []
    for doc in dense_docs + bm25_docs:
        key = (doc.metadata.get("source", "?"), doc.page_content[:80])
        if key not in seen:
            seen.add(key)
            pool.append(doc)
    if not pool:
        return [], []
    scores = reranker.predict([(query, doc.page_content) for doc in pool])
    ranked = sorted(zip(pool, scores), key=lambda pair: pair[1], reverse=True)[:k]
    return [doc for doc, _ in ranked], [float(score) for _, score in ranked]


def search_source_paths(query: str, k: int = TOP_K) -> list[str]:
    vectorstore = get_vectorstore(build_embeddings())
    docs = vectorstore.similarity_search(query, k=k)
    return _extract_sources(docs)


@lru_cache(maxsize=1)
def build_llm():
    provider = os.getenv("LLM_PROVIDER", "cerebras").lower()
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    from langchain_cerebras import ChatCerebras
    return ChatCerebras(
        model=os.getenv("CEREBRAS_MODEL", "gemma-4-31b"),
        api_key=os.getenv("CEREBRAS_API_KEY"),
        max_retries=0,
    )


@lru_cache(maxsize=1)
def build_cerebras_fallback_llm():
    """primary Cerebras 키가 429일 때 먼저 시도할 두 번째(유료) 키. 없으면 스킵하고 바로 Haiku."""
    key = os.getenv("CEREBRAS_FALLBACK_API_KEY")
    if not key:
        return None
    from langchain_cerebras import ChatCerebras
    return ChatCerebras(model=os.getenv("CEREBRAS_MODEL", "gemma-4-31b"), api_key=key, max_retries=0)


@lru_cache(maxsize=1)
def build_fallback_llm():
    model = os.getenv("LLM_FALLBACK_MODEL", "claude-haiku-4-5")
    if not model:
        return None
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=2)


@lru_cache(maxsize=1)
def build_judge_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=os.getenv("LLM_JUDGE_MODEL", "claude-sonnet-5"),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_retries=0,
    )


@lru_cache(maxsize=1)
def default_llm():
    return apply_fallback(build_llm())


def apply_fallback(model, configure=lambda m: m, fallback=None):
    if not isinstance(model, BaseChatModel):
        return configure(model)
    fb = fallback if fallback is not None else [build_cerebras_fallback_llm(), build_fallback_llm()]
    raw_models = [model] + [f for f in (fb if isinstance(fb, list) else [fb]) if f is not None]
    if len(raw_models) == 1:
        return configure(model)
    chain = [(getattr(m, "model", getattr(m, "model_name", type(m).__name__)), configure(m)) for m in raw_models]

    def _invoke_with_fallback(input):
        last_exc = None
        for i, (name, m) in enumerate(chain):
            try:
                result = m.invoke(input)
                if i > 0:
                    logger.warning("LLM 폴백 성공: %s (앞서 %d개 실패, 마지막 에러: %s)", name, i, last_exc)
                return result
            except Exception as exc:
                logger.warning("LLM 호출 실패, 다음으로 폴백: %s -> %s: %s", name, type(exc).__name__, exc)
                last_exc = exc
        raise last_exc

    return RunnableLambda(_invoke_with_fallback)


def answer_document_question(question: str, document_text: str, history: list[dict]) -> str:
    llm = apply_fallback(build_llm())
    chain = DOC_QA_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "document": document_text,
        "conversation": format_doc_chat_history(history),
        "question": question,
    })


def generate_recall_question(llm, topic: str, conversation: str) -> str:
    messages = RECALL_QUESTION_PROMPT.invoke(
        {"conversation": conversation, "topic": topic}
    ).to_messages()
    return apply_fallback(llm).invoke(messages).content


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
    embeddings = build_embeddings()
    vectorstore = get_vectorstore(embeddings)
    print(f"인덱싱 완료: {vectorstore._collection.count()}개 청크")


if __name__ == "__main__":
    ingest()
