from langgraph.graph import START, StateGraph, END
from state import GraphState
from rag import build_embeddings, build_llm, get_retriever, PROMPT
from nodes import make_nodes


def build_rag_graph():
    embeddings = build_embeddings()
    retriever = get_retriever(embeddings)
    llm = build_llm()

    retrieve_node, generate_node = make_nodes(retriever, PROMPT, llm)

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()