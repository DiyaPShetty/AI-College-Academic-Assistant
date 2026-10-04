from agents.graph import build_graph


graph = build_graph()


def run_agent(
    user_query,
    current_plan="",
    conversation_history=None
):

    if conversation_history is None:
        conversation_history = []

    state = {
        "user_query": user_query,
        "conversation_history": conversation_history,

        "intent": "",
        "rewritten_query": "",
        "needs_clarification": False,

        "subjects": [],
        "duration_days": 0,
        "hours_per_day": 0,
        "exam_date": "",

        "retrieved_docs": [],
        "retrieval_scores": [],
        "retrieval_relevant": False,

        "tool_result": "",

        "answer": "",
        "study_plan": current_plan,
        "modification": "",

        "review_status": "",
        "review_feedback": ""
    }

    return graph.invoke(state)


if __name__ == "__main__":

    result = run_agent(
        "What is the minimum attendance requirement?"
    )

    print("\n===== INTENT =====")
    print(result["intent"])

    print("\n===== ANSWER =====")
    print(result["answer"])
