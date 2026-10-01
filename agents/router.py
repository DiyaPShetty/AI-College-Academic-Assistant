from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


def route_query(state):

    query = state["user_query"]

    llm = get_llm()

    prompt = f"""
Classify the student's query into exactly ONE of these categories:

ACADEMIC
STUDY_PLAN
GENERAL

ACADEMIC:
Questions about college syllabus, subjects, academic regulations,
attendance, exams, courses, credits, internship rules, etc.

STUDY_PLAN:
Requests to create, modify, update, move, remove, or rearrange
a study plan.

GENERAL:
Anything that does not belong to the above categories.

Student query:
{query}

Return ONLY one word:
ACADEMIC
STUDY_PLAN
GENERAL
"""

    response = llm.invoke(prompt)

    intent = response.content.strip().upper()

    if intent not in ["ACADEMIC", "STUDY_PLAN", "GENERAL"]:
        intent = "GENERAL"

    return {
        **state,
        "intent": intent
    }