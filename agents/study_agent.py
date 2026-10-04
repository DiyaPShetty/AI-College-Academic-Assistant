from langchain_groq import ChatGroq
from dotenv import load_dotenv

from planner.study_planner import create_study_plan
from planner.plan_modifier import modify_study_plan

load_dotenv()


def extract_plan_parameters(query):
    """
    Extract subject, duration_days, and hours_per_day from user query.
    """
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = f"""
Extract the study plan parameters from the student's request.

Student request:
{query}

Extract:
1. subject: The subject/course name (e.g., DBMS, Data Structures, Python)
2. duration_days: Number of days for the study plan (integer)
3. hours_per_day: Hours to study per day (integer)

If a parameter is not mentioned, use these defaults:
- subject: DBMS
- duration_days: 7
- hours_per_day: 2

Return ONLY a JSON object with these exact keys:
{{
    "subject": "...",
    "duration_days": ...,
    "hours_per_day": ...
}}

Do not include any other text.
"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    try:
        import json
        params = json.loads(content)
        subject = params.get("subject", "DBMS")
        duration_days = int(params.get("duration_days", 7))
        hours_per_day = int(params.get("hours_per_day", 2))
    except (json.JSONDecodeError, ValueError):
        # Fallback to defaults if parsing fails
        subject = "DBMS"
        duration_days = 7
        hours_per_day = 2

    return subject, duration_days, hours_per_day


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
    # Extract parameters from user query
    subject, duration_days, hours_per_day = extract_plan_parameters(query)

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
