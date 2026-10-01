from typing import TypedDict


class AgentState(TypedDict):
    user_query: str
    intent: str
    retrieved_docs: list
    answer: str
    study_plan: str
    modification: str