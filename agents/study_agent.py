from planner.study_planner import create_study_plan
from planner.plan_modifier import modify_study_plan


def study_agent(state):

    query = state["user_query"]
    current_plan = state.get("study_plan", "")

    # If there is already a study plan,
    # treat the new request as a modification.
    if current_plan.strip():

        updated_plan, documents = modify_study_plan(
            current_plan,
            query
        )

        return {
            **state,
            "study_plan": updated_plan,
            "answer": updated_plan,
            "retrieved_docs": documents
        }

    # Otherwise create a new study plan
    subject = "DBMS"
    duration_days = 7
    hours_per_day = 2

    plan, documents = create_study_plan(
        subject,
        duration_days,
        hours_per_day
    )

    return {
        **state,
        "study_plan": plan,
        "answer": plan,
        "retrieved_docs": documents
    }