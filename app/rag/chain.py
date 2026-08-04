import logging
import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core import catalog
from app.core.paths import CHROMA_DIR, PROJECT_ROOT
from app.schemas.rag import TurnResult

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_COLLECTION = "cs231n_v2"
EMBED_MODEL = os.getenv("EMBED_MODEL", "dragonkue/multilingual-e5-small-ko-v2")
TOP_K = 5
RELEVANCE_THRESHOLD = 0.50


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


def get_vectorstore(embeddings, collection: str = CHROMA_COLLECTION):
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


def get_retriever(embeddings, k: int = TOP_K):
    return get_vectorstore(embeddings).as_retriever(search_kwargs={"k": k})


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
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    from langchain_cerebras import ChatCerebras
    return ChatCerebras(
        model=os.getenv("CEREBRAS_MODEL", "gemma-4-31b"),
        api_key=os.getenv("CEREBRAS_API_KEY"),
        max_retries=0,
    )


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


def apply_fallback(model, configure=lambda m: m, fallback=None):
    """fallback 없으면 build_fallback_llm() 사용, 리스트면 순서대로 시도. 전부 실패 시
    마지막 에러를 던진다(langchain with_fallbacks는 첫 에러만 던져 원인 파악이 안 됨)."""
    if not isinstance(model, BaseChatModel):
        return configure(model)
    fb = fallback if fallback is not None else build_fallback_llm()
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


DOC_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Q&A assistant answering questions about ONE specific document. "
     "Answer using ONLY the document text below — do not rely on prior knowledge or infer beyond "
     "what is explicitly written. Match your response language to the question (Korean → Korean, "
     "English → English) in EVERY case, including when you cannot answer. If the document lacks "
     "sufficient information, say so clearly in the question's own language — for example, "
     "Korean: \"이 문서에 이 내용이 없습니다.\" / English: \"This document does not cover this.\" "
     "Never default to English when the question was asked in another language.\n\n"
     "DOCUMENT:\n{document}"),
    ("human", "{conversation}Q: {question}"),
])


def _format_doc_chat_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = [f"{'Q' if h['role'] == 'user' else 'A'}: {h['content']}" for h in history]
    return "\n".join(lines) + "\n"


def answer_document_question(question: str, document_text: str, history: list[dict]) -> str:
    llm = apply_fallback(build_llm())
    chain = DOC_QA_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "document": document_text,
        "conversation": _format_doc_chat_history(history),
        "question": question,
    })


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


TOPIC_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract the distinct study topics or concepts the user says they covered today, from the "
     "conversation below (each line prefixed with its speaker: human/ai/tool). List each topic "
     "once, using the user's own wording — do not rename or generalize it into different "
     "terminology. Do not explain any topic.\n\n"
     "CRITICAL: a topic the human only ASKED ABOUT earlier in this conversation (a question "
     "answered via a tool/RAG lookup) is NOT something they studied today — only extract topics "
     "from what the human explicitly states THEY studied/covered themselves (a retrospective "
     "statement, e.g. \"오늘 도커 공부했어\"). If the human asked a question and got an answer, "
     "that exchange alone does not count unless the human separately says they studied it.\n\n"
     "Also write `umbrella`: a short title (a few words) covering all of today's topics together, "
     "used as this session's note title/filename. Prefer the broader subject they belong to over a "
     "comma-joined list of the topics themselves — e.g. topics `Dockerfile 캐시 레이어`, "
     "`bind-mount` → umbrella `Docker 배포`. Match the language of the conversation."),
    ("human", "{conversation}"),
])


TURN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are running a Korean-language retrieval-practice (인출 학습) study session. The user "
     "just tried to explain `{topic}` in their own words. Do TWO independent jobs and return them "
     "in the TurnResult schema.\n\n"
     "## JOB 1 — verdict (judge STANCE, never correctness)\n"
     "Classify the user's message into exactly one label. You NEVER judge whether the explanation "
     "is factually right — only the user's own expressed confidence.\n"
     "- \"explained\": a genuine, self-assured explanation in their own words.\n"
     "- \"partial\": they attempted an explanation but signaled THEIR OWN uncertainty — hedging "
     "(\"~인 것 같은데\", \"아마\", \"맞나?\"), trailing off, or saying part is unclear. The content "
     "may even be fully correct; what matters is that THEY are unsure.\n"
     "- \"skip\": declined or said they don't know, with no real attempt (\"모르겠어\", \"넘어가\", "
     "\"패스\", \"몰라\").\n\n"
     "## JOB 2 — stay_on_topic + next_question\n"
     "Check whether the user's message only NAMES a sub-point without truly explaining it — "
     "mentions a term or a split into parts but doesn't elaborate on any of them, glosses over "
     "the reasoning, or never says how it's actually used.\n\n"
     "This check is MANDATORY every single time, with NO exception for the last topic. Whether "
     "next_topic is a real topic or \"none\" has NO bearing on whether you run this check — "
     "decide shallow-vs-thorough first, from the content of the message ALONE. Being the last "
     "topic is never a reason to skip it or wrap up early. This pattern applies to ANY subject, "
     "not just technical ones.\n\n"
     "Example: user says \"광합성은 명반응이랑 암반응으로 나뉘어\" and stops there → shallow, "
     "stay_on_topic=true, ask a follow-up about what each reaction actually does. If they go on "
     "to also explain what happens in the 명반응 and what happens in the 암반응 → already "
     "thorough, stay_on_topic=false, move on.\n\n"
     "- If shallow: set stay_on_topic to true, and ask ONE follow-up question using exactly ONE "
     "of these three angles — whichever fits what was actually left unexplained:\n"
     "  1. Clarification: a question that pins down exactly what the mentioned part means or "
     "does.\n"
     "  2. Reasoning: a question asking why it works that way, or why it's needed.\n"
     "  3. Application: a question asking how it's actually used, or how it connects to another "
     "concept.\n"
     "  HARD CONSTRAINT: build the follow-up ONLY from words, terms, or facts the user has "
     "already said themselves earlier in this conversation. NEVER introduce a new term, a new "
     "comparison, or any fact the user has not already stated — doing so both pushes the "
     "difficulty past what they've shown they know, and can leak the answer through the "
     "question's own premise.\n"
     "- Otherwise (already thorough): set stay_on_topic to false.\n"
     "  - If next_topic is a real topic: write exactly ONE short, natural Korean question asking "
     "the user to explain `{next_topic}` in their own words.\n"
     "  - If next_topic is \"none\": the session is ending — set next_question to null. Do NOT "
     "invent a question.\n\n"
     "CRITICAL (applies to any question, whichever branch): NEVER explain, define, hint at, or "
     "reveal any part of the answer — the user must recall it themselves. Match the tone and "
     "flow of the conversation so far; vary your wording every time, never reuse a fixed "
     "template."),
    ("human",
     "next_topic: {next_topic}\n\nConversation so far:\n{conversation}\n\n"
     "User's message about `{topic}`: {message}"),
])


RECALL_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are running a retrieval-practice (인출 학습) study session. Ask the user to explain "
     "the given topic in their OWN words. Write ONE short, natural question that fits the tone "
     "and flow of the conversation so far — vary your wording, never reuse a fixed template. "
     "Output only the question, nothing else. "
     "CRITICAL: never explain, define, hint at, or reveal any part of the answer — the whole "
     "point is that the user recalls it themselves."),
    ("human", "Conversation so far:\n{conversation}\n\nAsk about this topic: {topic}"),
])


def generate_recall_question(llm, topic: str, conversation: str) -> str:
    messages = RECALL_QUESTION_PROMPT.invoke(
        {"conversation": conversation, "topic": topic}
    ).to_messages()
    return apply_fallback(llm).invoke(messages).content


AGENT_SYSTEM_PROMPT = (
    "You are a study coach. You have five tools: `answer_question`, `write_daily`, "
    "`write_weekly`, `write_til`, and `visualize_mindmap`. Rely on each tool's own description "
    "for when to use it.\n\n"
    "MUST — questions: whenever the user asks what something is or how it works, call "
    "`answer_question`. Never answer from your own knowledge, even if you are sure.\n\n"
    "MUST — retrospectives: whenever the user wants a project/task retrospective, call "
    "`write_til` (or `write_weekly` for a weekly summary). Never invent field values — if a "
    "required field is missing, ask the user for it.\n\n"
    "MUST — daily study: whenever the user wants to record or review what they studied today "
    "(회고형 발화, e.g. \"오늘 배운 거 정리해줘\"), call `write_daily` immediately. It starts a "
    "retrieval-practice (인출 학습) session that asks the user to recall each topic in their own "
    "words. Do NOT list topics, ask questions, or explain anything yourself — the session "
    "handles all of that. Your only job here is to call the tool.\n\n"
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
    embeddings = build_embeddings()
    vectorstore = get_vectorstore(embeddings)
    print(f"인덱싱 완료: {vectorstore._collection.count()}개 청크")


def build_rag_chain():
    embeddings = build_embeddings()
    retriever = get_retriever(embeddings)
    llm = apply_fallback(build_llm())

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
