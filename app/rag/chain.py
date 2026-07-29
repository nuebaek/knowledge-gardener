import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core import catalog
from app.core.paths import CHROMA_DIR, PROJECT_ROOT
from app.schemas.rag import TurnResult

load_dotenv()

CHROMA_COLLECTION = "cs231n"
TOP_K = 5
RELEVANCE_THRESHOLD = 0.4


def study_turn(llm, topic, next_topic, message, conversation) -> TurnResult:
    """인출 세션 한 턴을 LLM 한 번으로 처리한다 — 현재 답변 판정 + 다음 질문 생성.
    next_topic이 없으면(마지막 토픽) "none"을 넘겨 next_question을 null로 받는다."""
    turn = llm.with_structured_output(TurnResult)
    messages = TURN_PROMPT.invoke({
        "topic": topic,
        "next_topic": next_topic or "none",
        "message": message,
        "conversation": conversation,
    }).to_messages()
    return turn.invoke(messages)

# ---------------- 임베딩 & 벡터스토어 ----------------

@lru_cache(maxsize=1)
def build_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

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
        path = PROJECT_ROOT / row["source_path"]
        if not path.exists():
            continue  # 카탈로그엔 있는데 파일이 지워진 경우 — sync_index()와 같은 가드
        split_docs.extend(_split_path(path, header_splitter, char_splitter))
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


def search_source_paths(query: str, k: int = TOP_K) -> list[str]:
    """query와 관련된 코퍼스 문서의 경로만 돌려준다(생성 없이 검색만) — 🌱 토픽에 근거 위치를
    붙일 때 쓴다. qa_graph(풀 RAG)는 seedling마다 LLM 생성까지 돌아 낭비라 안 쓴다."""
    vectorstore = get_vectorstore(build_embeddings())
    docs = vectorstore.similarity_search(query, k=k)
    return _extract_sources(docs)


# ---------------- LLM ----------------

@lru_cache(maxsize=1)
def build_llm():
    """생성 LLM만 provider를 고른다(LLM_PROVIDER).

    lru_cache를 건 이유: writer·visualizer·tools가 각자 build_llm()을 부르면서 같은 설정의
    클라이언트를 여러 벌 만들고 있었다. provider 설정은 프로세스 수명 동안 안 바뀌므로 한 벌만
    만든다(테스트에서 다른 LLM이 필요하면 build_agent_graph/make_tools에 주입한다).
    """
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
        # Cerebras 기본 max_retries=None(재시도 없음) — rate limit(429)이 종종 나서 낮게 균일화.
        # langchain 표준 exponential backoff. 값을 낮게 둬 인터랙티브 호출 대기도 짧게 유지.
        max_retries=2,
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


TOPIC_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract the distinct study topics or concepts the user says they covered today, from the "
     "conversation below. List each topic once, using the user's own wording — do not rename or "
     "generalize it into different terminology. Do not explain any topic."),
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
    """RECALL_QUESTION_PROMPT로 대화 맥락에 맞는 인출 질문을 생성. study_node·write_daily가 공유."""
    messages = RECALL_QUESTION_PROMPT.invoke(
        {"conversation": conversation, "topic": topic}
    ).to_messages()
    return llm.invoke(messages).content


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
