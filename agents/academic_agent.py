from llm.rag_chain import ask_academic_assistant


def academic_agent(state):

    query = state["user_query"]

    answer, documents = ask_academic_assistant(query)

    return {
        **state,
        "answer": answer,
        "retrieved_docs": documents
    }