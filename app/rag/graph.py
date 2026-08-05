from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from app.rag.state import GraphState, StudySessionState
from app.rag.chain import build_embeddings, build_llm, get_vectorstore, search_source_paths, PROMPT, REWRITE_PROMPT, AGENT_SYSTEM_PROMPT, build_bm25, load_split_docs, build_reranker
from app.rag.nodes import make_nodes, make_agent_node, make_study_node


def route_after_grade(state: GraphState):
    if state.get("is_relevant", True):
        return "generate"
    if state.get("retry_count", 0) >= 2:
        return "generate"
    return "rewrite_query"


def check_pending(state: StudySessionState):
    if state.get("awaiting_finalize"):
        return "confirm_finalize"
    if state.get("awaiting_topic_confirm"):
        return "confirm_topics"
    return "study" if state.get("pending") else "agent"


def route_after_tools(state: StudySessionState):
    return END if state.get("pending") else "agent"


def build_rag_graph():
    embeddings = build_embeddings()
    vectorstore = get_vectorstore(embeddings)
    llm = build_llm()

    # dense+BM25 후보를 cross-encoder로 재점수한다(RRF는 n=210 실측에서 기각됨,
    # docs/2026-08-04-hybrid-search-design.md §4/§7) — 인덱스·리랭커는 그래프 빌드 시
    # 한 번만 만들고(코퍼스가 작아 매 요청 재구축은 낭비), 이후 매 질문마다 재사용.
    bm25_docs = load_split_docs()
    bm25 = build_bm25(bm25_docs)
    reranker = build_reranker()

    retrieve_node, generate_node, grade_docs_node, rewrite_query_node = make_nodes(
        vectorstore, PROMPT, REWRITE_PROMPT, llm, bm25=bm25, bm25_docs=bm25_docs, reranker=reranker,
    )

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_conditional_edges(
        "grade_docs",
        route_after_grade,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


def build_agent_graph(llm=None, tools=None, checkpointer=None, search_sources=None, judge_llm=None):
    from app.rag.tools import make_tools  # 순환 import 방지

    llm = llm if llm is not None else build_llm()
    tools = tools if tools is not None else make_tools(llm=llm)
    search_sources = search_sources if search_sources is not None else search_source_paths
    # 스터디 세션 판정은 생성 LLM과 같은 흐름(Cerebras → Haiku 폴백)을 쓴다 — Sonnet은
    # 오프라인 벤치마크(scripts/benchmark.py 등)의 judge 전용, 실사용 경로엔 안 태운다.
    judge_llm = judge_llm if judge_llm is not None else llm
    agent_node = make_agent_node(llm, tools, AGENT_SYSTEM_PROMPT)
    study_node, finalize_node, confirm_finalize_node, confirm_topics_node = make_study_node(judge_llm, llm, search_sources)

    graph = StateGraph(StudySessionState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("study", study_node)
    graph.add_node("confirm_finalize", confirm_finalize_node)
    graph.add_node("confirm_topics", confirm_topics_node)

    graph.add_conditional_edges(START, check_pending, {
        "agent": "agent", "study": "study",
        "confirm_finalize": "confirm_finalize", "confirm_topics": "confirm_topics",
    })
    graph.add_edge("study", END)
    graph.add_edge("confirm_finalize", END)
    graph.add_edge("confirm_topics", END)
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_conditional_edges("tools", route_after_tools, {END: END, "agent": "agent"})

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
