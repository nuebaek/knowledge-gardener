from rag import _extract_sources
from state import GraphState


def make_nodes(retriever, prompt, llm):
    def retrieve_node(state: GraphState):
        question = state["question"]
        return {"document": retriever.invoke(question)}

    def generate_node(state: GraphState):
        context = "\n\n".join(d.page_content for d in state["document"])
        messages = prompt.invoke({"context": context, "question": state["question"]}).to_messages()
        response = llm.invoke(messages)
        return {
            "answer": response.content,
            "sources": _extract_sources(state["document"]),
        }

    return retrieve_node, generate_node
