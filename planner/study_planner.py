from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# =========================================================
# LLM
# =========================================================

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=4096
    )


# =========================================================
# VECTOR STORE
# =========================================================

def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory="chroma_db",
        collection_name="college_knowledge",
        embedding_function=embeddings
    )


# =========================================================
# SYLLABUS RETRIEVAL
# =========================================================

def get_syllabus_context(subjects):

    if isinstance(subjects, str):
        subjects = [subjects]

    vectorstore = get_vectorstore()

    all_documents = []
    seen = set()

    for subject in subjects:

        if subject.lower() == "dbms":

            queries = [
                "DATABASE MANAGEMENT SYSTEMS CS2102-1 UNIT-I",
                "DATABASE MANAGEMENT SYSTEMS CS2102-1 UNIT-II",
                "DATABASE MANAGEMENT SYSTEMS CS2102-1 UNIT-III"
            ]

        else:

            queries = [
                f"{subject} syllabus UNIT-I UNIT-II UNIT-III",
                f"{subject} course contents topics"
            ]

        for query in queries:

            documents = vectorstore.similarity_search(
                query,
                k=4,
                filter={
                    "document_type": "department_syllabus"
                }
            )

            for document in documents:

                content = document.page_content.strip()

                if not content:
                    continue

                if subject.lower() == "dbms":

                    lower_content = content.lower()

                    if "is1601-1" in lower_content:
                        continue

                    if "database applications" in lower_content:
                        continue

                if content in seen:
                    continue

                seen.add(content)
                all_documents.append(document)

    # Limit context size
    all_documents = all_documents[:8]

    context_parts = []

    for document in all_documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        content = document.page_content.strip()

        if len(content) > 3500:
            content = content[:3500]

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

{content}
"""
        )

    context = "\n\n".join(context_parts)

    return context, all_documents


# =========================================================
# EXTRACT STUDY PLAN TABLE
# =========================================================

def extract_plan_table(answer):

    if not answer:
        return ""

    answer = str(answer).strip()

    # Remove markdown code fences
    answer = answer.replace("```markdown", "")
    answer = answer.replace("```md", "")
    answer = answer.replace("```", "")

    answer = answer.strip()

    # Find lines that look like table rows
    lines = [
        line.strip()
        for line in answer.splitlines()
        if line.strip().startswith("|")
    ]

    if not lines:
        return ""

    # Find the table header
    header_index = None

    for i, line in enumerate(lines):

        normalized = (
            line.lower()
            .replace(" ", "")
            .replace("\t", "")
        )

        if (
            "day" in normalized
            and "subject" in normalized
            and "unit" in normalized
            and "topics" in normalized
            and "duration" in normalized
            and "studygoal" in normalized
        ):

            header_index = i
            break

    if header_index is None:
        return ""

    table_lines = lines[header_index:]

    if len(table_lines) < 2:
        return ""

    # Standard header
    header = (
        "| Day | Subject | Unit | Topics | Duration | Study Goal |"
    )

    # Extract actual data rows
    data_rows = []

    for line in table_lines[1:]:

        cells = [
            cell.strip()
            for cell in line.strip("|").split("|")
        ]

        if len(cells) < 6:
            continue

        # Ignore separator row
        if all(
            "-" in cell
            for cell in cells
        ):
            continue

        # First column must be a day number
        try:
            int(cells[0])
        except ValueError:
            continue

        cells = cells[:6]

        normalized_row = (
            "| "
            + " | ".join(cells)
            + " |"
        )

        data_rows.append(normalized_row)

    if not data_rows:
        return ""

    separator = (
        "|-----|---------|------|--------|----------|------------|"
    )

    table = "\n".join(
        [header, separator] + data_rows
    )

    return table


# =========================================================
# CREATE STUDY PLAN
# =========================================================

def create_study_plan(
    subjects,
    duration_days,
    hours_per_day
):

    if isinstance(subjects, str):
        subjects = [subjects]

    if not subjects:

        return (
            "No subject was specified.",
            []
        )

    if duration_days <= 0:

        return (
            "The number of study days must be greater than zero.",
            []
        )

    if hours_per_day <= 0:

        return (
            "Study hours per day must be greater than zero.",
            []
        )

    # -----------------------------------------------------
    # Retrieve official syllabus
    # -----------------------------------------------------

    syllabus_context, documents = get_syllabus_context(
        subjects
    )

    if not syllabus_context.strip():

        return (
            "I could not find the requested subject in "
            "the official department syllabus.",
            documents
        )

    subject_text = ", ".join(subjects)

    # -----------------------------------------------------
    # Planner prompt
    # -----------------------------------------------------

    prompt = f"""
You are the study-planning component of a college
academic assistant.

Create a personalized {duration_days}-day study plan.

SUBJECTS:
{subject_text}

AVAILABLE STUDY TIME:
{hours_per_day} hours per day

OFFICIAL DEPARTMENT SYLLABUS:

{syllabus_context}


STRICT RULES:

1. Use ONLY the supplied official syllabus.

2. Do not use general knowledge.

3. Do not invent academic topics.

4. Do not use topics from another course.

5. For DBMS, use DATABASE MANAGEMENT SYSTEMS
   course CS2102-1.

6. Do not use DATABASE APPLICATIONS lab
   IS1601-1 content.

7. Use the actual syllabus units.

8. Do not use Course Objectives as study topics.

9. Do not use Course Outcomes as study topics.

10. Do not use textbook names as topics.

11. Do not repeat the same topic unless it is
    explicitly a review session.

12. Distribute related syllabus topics across
    the available study days.

13. Cover foundational material before later material
    where appropriate.

14. Create exactly {duration_days} days.

15. Every day must contain exactly
    {hours_per_day} hours.

16. Every topic must be traceable to the supplied
    syllabus.

17. Keep Topics concise.

18. Keep Study Goal concise.


OUTPUT FORMAT:

Return ONLY a markdown table.

The table MUST have exactly these columns:

| Day | Subject | Unit | Topics | Duration | Study Goal |
|-----|---------|------|--------|----------|------------|

Create exactly {duration_days} data rows.

Do not write any explanation before or after the table.
"""

    # -----------------------------------------------------
    # Call LLM
    # -----------------------------------------------------

    try:

        response = get_llm().invoke(prompt)

        answer = response.content

        if isinstance(answer, list):

            answer = "\n".join(
                str(item)
                for item in answer
            )

        answer = str(answer).strip()

    except Exception as exc:

        error_text = str(exc)

        if (
            "413" in error_text
            or "Request too large" in error_text
        ):

            return (
                "The study-plan request was too large for "
                "the current LLM request limit. Please try "
                "again with fewer subjects or fewer study days.",
                documents
            )

        raise

    # -----------------------------------------------------
    # Basic response check
    # -----------------------------------------------------

    if not answer:

        return (
            "The study planner did not return a valid plan.",
            documents
        )

    # -----------------------------------------------------
    # Extract clean table
    # -----------------------------------------------------

    table = extract_plan_table(answer)

    if not table:

        return (
            "The study planner returned an invalid plan format.",
            documents
        )

    # -----------------------------------------------------
    # Check number of generated days
    # -----------------------------------------------------

    table_lines = [
        line.strip()
        for line in table.splitlines()
        if line.strip().startswith("|")
    ]

    data_row_count = max(
        0,
        len(table_lines) - 2
    )

    if data_row_count != duration_days:

        return (
            f"The study planner generated "
            f"{data_row_count} days instead of "
            f"{duration_days}.",
            documents
        )

    # -----------------------------------------------------
    # Return clean table
    # -----------------------------------------------------

    return table, documents
