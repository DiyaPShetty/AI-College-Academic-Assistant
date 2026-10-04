from agents.graph import build_graph


graph = build_graph()


def run_agent(
    user_query,
    current_plan="",
    conversation_history=None
):
    """
    Run the academic assistant agent to process a user query.

    This function initializes the agent state with the user's query,
    conversation history, and current study plan (if any), then
    invokes the LangGraph workflow to generate a response.

    Args:
        user_query (str): The student's question or request.
        current_plan (str, optional): Existing study plan in markdown
            table format. Defaults to empty string.
        conversation_history (list, optional): List of previous
            conversation messages. Each message should be a dict with
            'role' and 'content' keys. Defaults to empty list.

    Returns:
        dict: The final agent state containing:
            - answer (str): The generated response.
            - intent (str): Classified intent (ACADEMIC, STUDY_PLAN, GENERAL, TOOL).
            - study_plan (str): Generated or modified study plan.
            - retrieved_docs (list): Retrieved documents for RAG.
            - review_status (str): Response validation status.
            - review_feedback (str): Validation feedback details.
    """
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
