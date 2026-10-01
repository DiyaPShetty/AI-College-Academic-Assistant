from agents.graph import build_graph


graph = build_graph()


def run_agent(user_query, current_plan=""):

    state = {
        "user_query": user_query,
        "intent": "",
        "retrieved_docs": [],
        "answer": "",
        "study_plan": current_plan,
        "modification": ""
    }

    result = graph.invoke(state)

    return result


if __name__ == "__main__":

    result = run_agent(
        "What is the minimum attendance required?"
    )

    print("\n===== INTENT =====")
    print(result["intent"])

    print("\n===== ANSWER =====")
    print(result["answer"])