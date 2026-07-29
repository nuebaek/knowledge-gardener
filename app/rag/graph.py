from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.memory import InMemorySaver

from app.rag.state import GraphState, StudySessionState
from app.rag.chain import build_embeddings, build_llm, get_vectorstore, search_source_paths, PROMPT, REWRITE_PROMPT, AGENT_SYSTEM_PROMPT
from app.rag.nodes import make_nodes, make_agent_node, make_study_node
from app.rag.study_session import is_complete


def route_after_grade(state: GraphState):
    """grade_docs 다음 분기: 관련 있으면 generate, 없으면 retry_count<2 한도 내에서 rewrite_query."""
    if state.get("is_relevant", True):
        return "generate"
    if state.get("retry_count", 0) >= 2:
        return "generate"
    return "rewrite_query"


def route(state: StudySessionState):
    """study 다음 분기: 남은 토픽 없으면 finalize(저장), 있으면 study가 넣은 질문으로 END."""
    return "finalize" if is_complete(state) else "end"


def check_pending(state: StudySessionState):
    """턴 진입 분기: 인출 세션이 진행 중이면 사용자의 말은 새 요청이 아니라 인출 질문의 답이다."""
    return "study" if state.get("pending") else "agent"


def route_after_tools(state: StudySessionState):
    """tool 실행 다음 분기. pending이 찼다 = write_daily가 방금 인출 세션을 열었다는 뜻 —
    그 질문(ToolMessage)이 agent를 안 거치고 그대로 나가야 힌트가 안 새므로 END로 보낸다.
    (goto=END가 아니라 조건 엣지인 이유: docs/2026-07-29-study-loop-design.md §3)
    """
    return END if state.get("pending") else "agent"


def build_rag_graph():
    embeddings = build_embeddings()
    vectorstore = get_vectorstore(embeddings)
    llm = build_llm()

    retrieve_node, generate_node, grade_docs_node, rewrite_query_node = make_nodes(vectorstore, PROMPT, REWRITE_PROMPT, llm)

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


def build_agent_graph(llm=None, tools=None, checkpointer=None, search_sources=None):
    """llm/tools/checkpointer/search_sources를 주입 가능하게 둔 건 테스트를 위해서다 —
    기본값은 예전과 똑같이 실제 LLM·tool 목록·코퍼스 검색을 쓴다."""
    from app.rag.tools import make_tools  # graph <-> tools 순환 import 방지

    llm = llm if llm is not None else build_llm()
    tools = tools if tools is not None else make_tools(llm=llm)
    search_sources = search_sources if search_sources is not None else search_source_paths
    agent_node = make_agent_node(llm, tools, AGENT_SYSTEM_PROMPT)
    study_node, finalize_node = make_study_node(llm, search_sources)

    graph = StateGraph(StudySessionState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("study", study_node)
    graph.add_node("finalize", finalize_node)

    graph.add_conditional_edges(START, check_pending, {"agent": "agent", "study": "study"})
    graph.add_conditional_edges("study", route, {"finalize": "finalize", "end": END})
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_conditional_edges("tools", route_after_tools, {END: END, "agent": "agent"})
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())
