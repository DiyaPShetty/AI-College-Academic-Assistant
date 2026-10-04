from typing import TypedDict


class AgentState(TypedDict):
    # User input
    user_query: str
    conversation_history: list

    # Question analysis
    intent: str
    rewritten_query: str
    needs_clarification: bool

    # Study planner parameters
    subjects: list
    duration_days: int
    hours_per_day: float
    exam_date: str

    # RAG
    retrieved_docs: list
    retrieval_scores: list
    retrieval_relevant: bool

    # Tool
    tool_result: str

    # Outputs
    answer: str
    study_plan: str
    modification: str

    # Review
    review_status: str
    review_feedback: str
