import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


def _extract_json(text: str) -> dict:
    text = text.strip()

    # Remove markdown code fences if the model adds them
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def route_query(state):
    query = state["user_query"]
    history = state.get("conversation_history", [])

    history_text = "\n".join(
        f"{m.get('role', '')}: {m.get('content', '')}"
        for m in history[-6:]
    )

    prompt = f"""
You are the question-analysis node of a college academic assistant.

Classify the student's current request.

Allowed intents:
ACADEMIC
STUDY_PLAN
GENERAL
TOOL

ACADEMIC:
Questions requiring college documents such as syllabus,
regulations, attendance, exams, credits, courses, internship
rules, academic policies, etc.

STUDY_PLAN:
Creating, modifying, updating, rearranging or removing
parts of a study plan.

TOOL:
Requests that require calculation, arithmetic, percentage,
conversion or another explicit computational operation.

GENERAL:
Normal conversation or questions outside the college
knowledge base that do not require the calculator.

Use the conversation history to understand follow-up questions.

Conversation history:
{history_text}

Current student request:
{query}

Return ONLY JSON:

{{
  "intent": "ACADEMIC",
  "rewritten_query": "standalone version of the student's question",
  "needs_clarification": false
}}

Rules:
- rewritten_query must preserve the student's actual intent.
- Resolve references such as "this", "that", "mine", "it",
  "day 5", etc. using conversation history when possible.
- Do not answer the question.
- Return exactly one intent.
"""

    response = get_llm().invoke(prompt)
    data = _extract_json(response.content)

    intent = str(data.get("intent", "GENERAL")).upper()

    if intent not in {"ACADEMIC", "STUDY_PLAN", "GENERAL", "TOOL"}:
        intent = "GENERAL"

    rewritten_query = data.get("rewritten_query") or query

    return {
        **state,
        "intent": intent,
        "rewritten_query": rewritten_query,
        "needs_clarification": bool(
            data.get("needs_clarification", False)
        )
    }
