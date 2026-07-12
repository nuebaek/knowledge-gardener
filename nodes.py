from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from rag import _extract_sources, TOP_K, RELEVANCE_THRESHOLD
from state import GraphState


def make_agent_node(llm, tools, system_prompt):
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent_node


def make_nodes(vectorstore, prompt, rewrite_prompt, llm):
    def retrieve_node(state: GraphState):
        rewritten_question = state.get("rewritten_question", "")
        if rewritten_question:
            question = rewritten_question
        else:
            question = state["question"]
        results = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
        return {
            "document": [doc for doc, _ in results],
            "doc_scores": [score for _, score in results],
        }

    def generate_node(state: GraphState):
        context = "\n\n".join(d.page_content for d in state["document"])
        messages = prompt.invoke({"context": context, "question": state["question"]}).to_messages()
        response = llm.invoke(messages)
        sources = _extract_sources(state["document"]) if state.get("is_relevant", True) else []
        return {
            "answer": response.content,
            "sources": sources,
        }
    
    def grade_docs_node(state: GraphState):
        top_score = max(state["doc_scores"])
        is_relevant = top_score > RELEVANCE_THRESHOLD
        print(f"[grade_docs] top_score={top_score:.3f} threshold={RELEVANCE_THRESHOLD} -> is_relevant={is_relevant}")
        return {"is_relevant": is_relevant}


    def rewrite_query_node(state: GraphState):
        current_query = state.get("rewritten_question") or state["question"]
        messages = rewrite_prompt.invoke({
            "question": state["question"],
            "rewritten_question": state.get("rewritten_question", ""),
        }).to_messages()
        response = llm.invoke(messages)
        print(f"[rewrite_query] attempt {state.get('retry_count', 0) + 1}: {current_query!r} -> {response.content!r}")
        return {
            "rewritten_question": response.content,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    return retrieve_node, generate_node, grade_docs_node, rewrite_query_node
