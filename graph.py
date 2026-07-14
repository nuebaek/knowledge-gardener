from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from state import GraphState
from rag import build_embeddings, build_llm, get_vectorstore, PROMPT, REWRITE_PROMPT, AGENT_SYSTEM_PROMPT
from nodes import make_nodes, make_agent_node
from langgraph.checkpoint.memory import InMemorySaver

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


def build_agent_graph():
    from tools import make_tools 

    llm = build_llm()
    tools = make_tools()
    agent_node = make_agent_node(llm, tools, AGENT_SYSTEM_PROMPT)

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=InMemorySaver())
