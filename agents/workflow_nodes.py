from datetime import date
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag.retriever import retrieve_with_scores
from llm.rag_chain import generate_academic_answer
from planner.study_planner import create_study_plan
from planner.plan_modifier import modify_study_plan


load_dotenv()


# =========================================================
# KNOWN SUBJECTS
# =========================================================

KNOWN_SUBJECTS = {
    "dbms": "DBMS",
    "database management systems": "DBMS",

    "data structures": "Data Structures",
    "data structure": "Data Structures",

    "operating systems": "Operating Systems",
    "operating system": "Operating Systems",

    "computer networks": "Computer Networks",

    "data communication and networking":
        "Data Communication and Networking",

    "data communication networking":
        "Data Communication and Networking",

    "image processing": "Image Processing",

    "design and analysis of algorithms":
        "Design and Analysis of Algorithms",

    "daa": "Design and Analysis of Algorithms",

    "object oriented programming":
        "Object Oriented Programming",

    "oop": "Object Oriented Programming",

    "java": "Java",

    "python": "Python",
}


# =========================================================
# DETERMINISTIC STUDY-PLAN EXTRACTION HELPERS
# =========================================================

def detect_subjects_from_query(query):
    """
    Detect known subjects directly from the user's query.

    This acts as a fallback if the LLM fails to extract
    the subject correctly.
    """

    query_lower = query.lower()

    found = []

    aliases = sorted(
        KNOWN_SUBJECTS.keys(),
        key=len,
        reverse=True
    )

    for alias in aliases:

        pattern = (
            r"(?<!\w)"
            + re.escape(alias)
            + r"(?!\w)"
        )

        if re.search(pattern, query_lower):

            canonical = KNOWN_SUBJECTS[alias]

            if canonical not in found:
                found.append(canonical)

    return found


def detect_duration_from_query(query):
    """
    Detect expressions such as:

    7 days
    10 days
    2 day
    """

    match = re.search(
        r"\b(\d+)\s*(?:day|days)\b",
        query.lower()
    )

    if match:
        return int(match.group(1))

    return None


def detect_hours_from_query(query):
    """
    Detect expressions such as:

    2 hours
    1 hour
    2 hrs
    1.5 hours
    2 hours per day
    """

    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*"
        r"(?:hour|hours|hr|hrs)"
        r"(?:\s*per\s*day)?\b",
        query.lower()
    )

    if match:
        return float(match.group(1))

    return None


# =========================================================
# LLM
# =========================================================

def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


# =========================================================
# RETRIEVAL NODE
# =========================================================

def retrieval_node(state):

    query = (
        state.get("rewritten_query")
        or state["user_query"]
    )

    documents, scores = retrieve_with_scores(
        query,
        k=5
    )

    # Chroma relevance threshold.
    relevant = bool(
        documents
        and scores
        and max(scores) >= 0.30
    )

    return {
        **state,
        "retrieved_docs": documents if relevant else [],
        "retrieval_scores": scores,
        "retrieval_relevant": relevant
    }


# =========================================================
# ACADEMIC GENERATION NODE
# =========================================================

def academic_generation_node(state):

    if not state.get("retrieval_relevant", False):

        return {
            **state,
            "answer": (
                "I couldn't find this information in the "
                "available college documents."
            )
        }

    answer = generate_academic_answer(
        state.get("rewritten_query")
        or state["user_query"],
        state["retrieved_docs"],
        state.get("conversation_history", [])
    )

    return {
        **state,
        "answer": answer
    }


# =========================================================
# STUDY PLAN PARAMETER EXTRACTION
# =========================================================

def extract_plan_parameters(query):

    prompt = f"""
Extract study-plan information from this student request.

Student request:

{query}

Return ONLY valid JSON:

{{
    "subjects": [],
    "duration_days": null,
    "hours_per_day": null,
    "exam_date": null
}}

Rules:

- subjects must be a list of subject/course names.
- duration_days must be an integer or null.
- hours_per_day must be a number or null.
- exam_date must use YYYY-MM-DD or null.
- Never invent missing values.
"""

    try:

        response = get_llm().invoke(prompt)

        text = response.content.strip()

        # Remove markdown code fences if the LLM adds them.
        text = re.sub(
            r"```json\s*|\s*```",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

        # Find JSON object.
        match = re.search(
            r"\{.*\}",
            text,
            re.DOTALL
        )

        if match:

            data = json.loads(match.group(0))

        else:

            data = {}

    except Exception:

        data = {}

    # -----------------------------------------------------
    # Values extracted by LLM
    # -----------------------------------------------------

    subjects = data.get("subjects") or []

    if isinstance(subjects, str):
        subjects = [subjects]

    duration = data.get("duration_days")

    hours = data.get("hours_per_day")

    exam_date = data.get("exam_date")

    # -----------------------------------------------------
    # Convert types safely
    # -----------------------------------------------------

    try:

        duration = (
            int(duration)
            if duration is not None
            else None
        )

    except (TypeError, ValueError):

        duration = None

    try:

        hours = (
            float(hours)
            if hours is not None
            else None
        )

    except (TypeError, ValueError):

        hours = None

    # =====================================================
    # DETERMINISTIC FALLBACK
    # =====================================================
    #
    # If the LLM missed something, extract it directly
    # from the user's query.
    #

    detected_subjects = detect_subjects_from_query(query)

    if not subjects and detected_subjects:

        subjects = detected_subjects

    detected_duration = detect_duration_from_query(query)

    if duration is None and detected_duration is not None:

        duration = detected_duration

    detected_hours = detect_hours_from_query(query)

    if hours is None and detected_hours is not None:

        hours = detected_hours

    # -----------------------------------------------------
    # Normalize subjects
    # -----------------------------------------------------

    normalized_subjects = []

    for subject in subjects:

        subject = str(subject).strip()

        if not subject:
            continue

        detected = KNOWN_SUBJECTS.get(
            subject.lower()
        )

        if detected:

            subject = detected

        if subject not in normalized_subjects:

            normalized_subjects.append(subject)

    subjects = normalized_subjects

    return (
        subjects,
        duration,
        hours,
        exam_date
    )


# =========================================================
# STUDY PLAN NODE
# =========================================================

def study_plan_node(state):

    query = state["user_query"]

    # Existing plan means modification request.
    current_plan = state.get("study_plan", "")

    if current_plan.strip():

        updated_plan = modify_study_plan(
            current_plan,
            query
        )

        return {
            **state,
            "study_plan": updated_plan,
            "answer": updated_plan,
            "retrieved_docs": []
        }

    # -----------------------------------------------------
    # Extract study-plan parameters
    # -----------------------------------------------------

    (
        subjects,
        duration,
        hours,
        exam_date
    ) = extract_plan_parameters(query)

    # -----------------------------------------------------
    # Calculate duration from exam date
    # -----------------------------------------------------

    if duration is None and exam_date:

        try:

            exam = date.fromisoformat(exam_date)

            today = date.today()

            duration = (exam - today).days

            if duration <= 0:

                duration = None

        except ValueError:

            duration = None

    # -----------------------------------------------------
    # Check missing information
    # -----------------------------------------------------

    missing = []

    if not subjects:

        missing.append("subject(s)")

    if duration is None:

        missing.append(
            "number of days or a future exam date"
        )

    if hours is None:

        missing.append(
            "available study hours per day"
        )

    if missing:

        message = (
            "I can create the study plan, but I need "
            + ", ".join(missing)
            + "."
        )

        return {
            **state,
            "subjects": subjects,
            "duration_days": duration or 0,
            "hours_per_day": hours or 0,
            "exam_date": exam_date or "",
            "needs_clarification": True,
            "answer": message
        }

    # -----------------------------------------------------
    # Generate plan
    # -----------------------------------------------------

    plan, documents = create_study_plan(
        subjects,
        duration,
        hours
    )

    return {
        **state,
        "subjects": subjects,
        "duration_days": duration,
        "hours_per_day": hours,
        "exam_date": exam_date or "",
        "study_plan": plan,
        "answer": plan,
        "retrieved_docs": documents
    }


# =========================================================
# STUDY PLAN VALIDATOR
# =========================================================

def plan_validator(state):
    """
    Validate the generated study plan against
    the requested parameters.
    """

    plan = state.get("study_plan", "")
    subjects = state.get("subjects", [])
    duration_days = state.get("duration_days", 0)
    hours_per_day = state.get("hours_per_day", 0)

    errors = []
    warnings = []

    # -----------------------------------------------------
    # Basic plan existence check
    # -----------------------------------------------------

    if not plan or "|" not in plan:

        errors.append(
            "Study plan was not generated as a valid table."
        )

    required_columns = [
        "Day",
        "Subject",
        "Unit",
        "Topics",
        "Duration",
        "Study Goal",
    ]

    # -----------------------------------------------------
    # Extract markdown table rows
    # -----------------------------------------------------

    lines = [
        line.strip()
        for line in plan.splitlines()
        if line.strip().startswith("|")
    ]

    if len(lines) < 3:

        errors.append(
            "Study plan does not contain enough table rows."
        )

    else:

        header = [
            x.strip()
            for x in lines[0].strip("|").split("|")
        ]

        # -------------------------------------------------
        # Column validation
        # -------------------------------------------------

        missing_columns = [
            column
            for column in required_columns
            if column not in header
        ]

        if missing_columns:

            errors.append(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        # -------------------------------------------------
        # Parse data rows
        # -------------------------------------------------

        data_rows = []

        for line in lines[2:]:

            cells = [
                x.strip()
                for x in line.strip("|").split("|")
            ]

            if len(cells) != len(header):

                warnings.append(
                    "One table row has an unexpected "
                    "number of columns."
                )

                continue

            row = dict(zip(header, cells))

            data_rows.append(row)

        # -------------------------------------------------
        # Day validation
        # -------------------------------------------------

        if duration_days:

            if len(data_rows) != duration_days:

                errors.append(
                    f"Expected {duration_days} study days, "
                    f"but generated {len(data_rows)}."
                )

        expected_days = list(
            range(1, len(data_rows) + 1)
        )

        actual_days = []

        for row in data_rows:

            try:

                actual_days.append(
                    int(row["Day"])
                )

            except (ValueError, KeyError):

                errors.append(
                    "Invalid Day value found "
                    "in study plan."
                )

        if actual_days and actual_days != expected_days:

            errors.append(
                f"Days are not sequential. "
                f"Expected {expected_days}, "
                f"got {actual_days}."
            )

        # -------------------------------------------------
        # Duration validation
        # -------------------------------------------------

        if hours_per_day:

            for row in data_rows:

                duration_text = row.get(
                    "Duration",
                    ""
                )

                match = re.search(
                    r"(\d+(?:\.\d+)?)",
                    duration_text
                )

                if match:

                    actual_hours = float(
                        match.group(1)
                    )

                    if abs(
                        actual_hours - hours_per_day
                    ) > 0.01:

                        errors.append(
                            f"Day {row.get('Day', '?')} "
                            f"has {actual_hours} hrs "
                            f"instead of "
                            f"{hours_per_day} hrs."
                        )

        # -------------------------------------------------
        # Subject validation
        # -------------------------------------------------

        if subjects:

            requested_subjects = {
                subject.lower().strip()
                for subject in subjects
            }

            for row in data_rows:

                subject = row.get(
                    "Subject",
                    ""
                ).lower().strip()

                if subject not in requested_subjects:

                    errors.append(
                        f"Unexpected subject "
                        f"'{row.get('Subject', '')}' "
                        f"found in Day "
                        f"{row.get('Day', '?')}."
                    )

        # -------------------------------------------------
        # Duplicate topic detection
        # -------------------------------------------------

        seen_topics = set()

        for row in data_rows:

            topic = row.get(
                "Topics",
                ""
            ).strip().lower()

            if not topic:

                errors.append(
                    f"Day {row.get('Day', '?')} "
                    f"has no topic."
                )

                continue

            if topic in seen_topics:

                warnings.append(
                    f"Repeated topic detected: "
                    f"{row.get('Topics', '')}"
                )

            seen_topics.add(topic)

        # -------------------------------------------------
        # DBMS-specific protection
        # -------------------------------------------------
        #
        # Prevent the previous Database Applications
        # laboratory-content problem from returning.
        #

        if any(
            subject.lower() == "dbms"
            for subject in subjects
        ):

            forbidden_terms = [
                "database applications",
                "is1601-1",
                "student database schema diagram",
                "employee database schema diagram",
                "insurance database schema diagram",
                "movie database schema diagram",
            ]

            plan_lower = plan.lower()

            for term in forbidden_terms:

                if term in plan_lower:

                    errors.append(
                        "DBMS plan contains "
                        "Database Applications "
                        f"lab content: '{term}'."
                    )

    # -----------------------------------------------------
    # Validation result
    # -----------------------------------------------------

    if errors:

        validation_status = "FAILED"

    elif warnings:

        validation_status = "PASSED_WITH_WARNINGS"

    else:

        validation_status = "PASSED"

    feedback = {
        "status": validation_status,
        "errors": errors,
        "warnings": warnings,
    }

    state["review_feedback"] = json.dumps(
        feedback,
        indent=2
    )

    if validation_status == "FAILED":

        state["review_status"] = "FAILED"

    else:

        state["review_status"] = "PASSED"

    return state


# =========================================================
# GENERAL NODE
# =========================================================

def general_generation_node(state):

    query = (
        state.get("rewritten_query")
        or state["user_query"]
    )

    history = "\n".join(
        f"{m.get('role')}: {m.get('content')}"
        for m in state.get(
            "conversation_history",
            []
        )[-6:]
    )

    prompt = f"""
You are a helpful college AI assistant.

Conversation:

{history}

Student question:

{query}

Answer clearly and concisely.
"""

    response = get_llm().invoke(prompt)

    return {
        **state,
        "answer": str(response.content).strip()
    }


# =========================================================
# CALCULATOR TOOL NODE
# =========================================================

def calculator_node(state):

    import requests

    query = state["user_query"]

    # Ask the LLM to extract the mathematical expression.
    prompt = f"""
Extract the mathematical expression from the following
student request.

Student request:

{query}

Return ONLY the expression.

Do not explain it.
"""

    expression_response = get_llm().invoke(prompt)

    expression = expression_response.content.strip()

    expression = expression.replace("`", "")
    expression = expression.replace("=", "")

    try:

        response = requests.get(
            "https://api.mathjs.org/v4/",
            params={
                "expr": expression
            },
            timeout=10
        )

        response.raise_for_status()

        result = response.text.strip()

        answer = (
            f"Calculation: {expression}\n\n"
            f"Result: {result}"
        )

    except Exception as exc:

        answer = (
            "I could not complete the calculation using "
            f"the calculator service. Error: {exc}"
        )

    return {
        **state,
        "tool_result": answer,
        "answer": answer
    }


# =========================================================
# RESPONSE REVIEW NODE
# =========================================================

def response_review_node(state):

    answer = state.get("answer", "")

    if not answer:

        return {
            **state,
            "review_status": "FAIL",
            "review_feedback": "Empty response."
        }

    # -----------------------------------------------------
    # Academic answer review
    # -----------------------------------------------------

    if state.get("intent") == "ACADEMIC":

        context = "\n\n".join(
            doc.page_content
            for doc in state.get(
                "retrieved_docs",
                []
            )
        )

        prompt = f"""
Review this academic assistant answer.

QUESTION:

{state.get("rewritten_query") or state["user_query"]}

ANSWER:

{answer}

SOURCE CONTEXT:

{context}

Check:

1. Is the answer supported by the source context?
2. Does it answer the actual question?
3. Does it invent college-specific facts?
4. Does it contradict the source?

Return ONLY:

PASS

or

FAIL: <short reason>
"""

        review = get_llm().invoke(prompt)

        result = review.content.strip()

        if result.upper().startswith("PASS"):

            return {
                **state,
                "review_status": "PASS",
                "review_feedback": result
            }

        return {
            **state,
            "review_status": "FAIL",
            "review_feedback": result
        }

    # -----------------------------------------------------
    # Other response types
    # -----------------------------------------------------

    return {
        **state,
        "review_status": "PASS",
        "review_feedback": "Basic validation passed."
    }