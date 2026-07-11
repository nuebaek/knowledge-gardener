from langgraph.graph import START, StateGraph, END
from state import GraphState
from rag import build_embeddings, build_llm, PROMPT, REWRITE_PROMPT, get_vectorstore
from nodes import make_nodes

def route_after_grade(state: GraphState):
    """grade_docs 다음 분기: 관련 있으면 generate, 없으면 retry_count<2 한도 내에서 rewrite_query."""
    if state.get("is_relevant", True):
        return "generate"
    if state.get("retry_count", 0) >= 2:
        return "generate"
    return "rewrite_query"

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
