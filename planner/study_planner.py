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
# Vector Store
# =========================================================

def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vectorstore


# =========================================================
# Retrieve Syllabus Context
# =========================================================

def get_syllabus_context(subject):

    vectorstore = get_vectorstore()

    # -----------------------------------------------------
    # DBMS-specific retrieval
    # -----------------------------------------------------

    if subject.upper() == "DBMS":

        results = vectorstore.similarity_search(
            "CS2102-1 DATABASE MANAGEMENT SYSTEMS",
            k=10
        )

        relevant_documents = []

        for document in results:

            page = document.metadata.get("page")
            content = document.page_content.upper()

            if (
                page in [128, 129]
                and (
                    "DATABASE MANAGEMENT SYSTEMS" in content
                    or "CS2102-1" in content
                    or "BASIC SQL" in content
                    or "STORAGE AND INDEXING" in content
                    or "TRANSACTION MANAGEMENT" in content
                )
            ):
                relevant_documents.append(document)

    # -----------------------------------------------------
    # General subject retrieval
    # -----------------------------------------------------

    else:

        relevant_documents = vectorstore.similarity_search(
            f"{subject} official syllabus",
            k=5
        )


    # =====================================================
    # Remove duplicate documents
    # =====================================================

    unique_documents = []

    seen = set()

    for document in relevant_documents:

        content = document.page_content.strip()

        if content not in seen:

            seen.add(content)

            unique_documents.append(document)


    # =====================================================
    # Sort documents by page
    # =====================================================

    unique_documents.sort(
        key=lambda document: document.metadata.get("page", 0)
    )


    # =====================================================
    # Keep only the most relevant documents
    # =====================================================

    documents = unique_documents[:3]


    # =====================================================
    # Combine document content
    # =====================================================

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    return context, documents


# =========================================================
# Create Study Plan
# =========================================================

def create_study_plan(
    subject,
    duration_days,
    hours_per_day
):

    # =====================================================
    # RETRIEVE SYLLABUS
    # =====================================================

    syllabus_context, documents = get_syllabus_context(subject)

    print("\n===== RETRIEVED SYLLABUS CONTEXT =====\n")
    print(syllabus_context)

    print("\n===== END CONTEXT =====\n")


    # =====================================================
    # CHECK RETRIEVED CONTEXT
    # =====================================================

    if not syllabus_context.strip():

        return (
            "No relevant syllabus information was retrieved.",
            documents
        )


    # =====================================================
    # GET LLM
    # =====================================================

    llm = get_llm()


    # =====================================================
    # MAIN PROMPT
    # =====================================================

    prompt = f"""
Create a {duration_days}-day study plan for {subject}.

Study time:
{hours_per_day} hours per day.

You MUST use ONLY the official syllabus content
provided below.

==================================================
OFFICIAL SYLLABUS
==================================================

{syllabus_context}

==================================================
STRICT RULES
==================================================

1. Use ONLY topics explicitly present in the
   official syllabus.

2. Preserve the exact terminology used in the
   official syllabus.

3. Preserve the official unit numbers and unit
   names exactly as they appear in the syllabus.

4. Do NOT create new units.

5. Do NOT rename units.

6. Do NOT change the official unit numbering.

7. Do NOT invent topics or subtopics.

8. Do NOT use outside knowledge.

9. Do NOT add textbooks.

10. Do NOT add study resources.

11. Do NOT add chapter numbers.

12. Do NOT add page numbers.

13. Do NOT add textbook references.

14. Do NOT add reference codes such as:
    T2: 8.2
    T2: 8.3
    Chapter 8
    Page 120

15. Do NOT add examples that are not explicitly
    present in the syllabus.

16. Every topic in the Topics column must come
    directly from the supplied syllabus.

17. The Study Goal must describe the topics
    assigned to that day without introducing
    new subject matter.

18. Keep exactly {hours_per_day} hours per day.

19. Cover the available syllabus topics across
    exactly {duration_days} days.

20. Do not add explanations before or after
    the table.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY a markdown table.

Use exactly these columns:

| Day | Unit | Topics | Duration | Study Goal |
|-----|------|--------|----------|------------|

The Unit column must preserve the official
syllabus unit information.

The Topics column must contain only syllabus
topics.

The Duration column must contain exactly
{hours_per_day} hours for each day.

The Study Goal must be based only on the topics
assigned to that day.
"""


    # =====================================================
    # CALL GROQ
    # =====================================================

    print("\n===== CALLING GROQ =====\n")

    response = llm.invoke(prompt)


    # =====================================================
    # DISPLAY RESPONSE OBJECT
    # =====================================================

    print("\n===== RESPONSE OBJECT =====")
    print(response)


    # =====================================================
    # EXTRACT RESPONSE
    # =====================================================

    answer = response.content


    if isinstance(answer, list):

        answer = "\n".join(
            str(item)
            for item in answer
        )


    answer = str(answer).strip()


    # =====================================================
    # DEBUG INFORMATION
    # =====================================================

    print("\n===== LLM RESPONSE LENGTH =====")
    print(len(answer))

    print("\n===== RAW LLM RESPONSE =====")
    print(repr(answer))


    # =====================================================
    # RETRY IF EMPTY
    # =====================================================

    if not answer:

        print(
            "\n===== EMPTY RESPONSE - RETRYING =====\n"
        )


        retry_prompt = f"""
Create a {duration_days}-day study plan for
{subject}.

Use ONLY the official syllabus below.

OFFICIAL SYLLABUS:

{syllabus_context}

Study time:
{hours_per_day} hours per day.

STRICT REQUIREMENTS:

- Use only topics explicitly written in the syllabus.
- Preserve the official unit names and numbering.
- Do not invent or rename units.
- Do not invent topics.
- Do not use outside knowledge.
- Do not add textbooks.
- Do not add resources.
- Do not add chapter numbers.
- Do not add page numbers.
- Do not add textbook references.
- Do not add codes such as T2: 8.2.
- Keep {hours_per_day} hours per day.
- Do not add explanations.

Return ONLY:

| Day | Unit | Topics | Duration | Study Goal |
|-----|------|--------|----------|------------|

Create exactly {duration_days} days.
"""


        retry_response = llm.invoke(
            retry_prompt
        )


        # -------------------------------------------------
        # Display retry response
        # -------------------------------------------------

        print("\n===== RETRY RESPONSE =====")
        print(retry_response)


        # -------------------------------------------------
        # Extract retry response
        # -------------------------------------------------

        answer = retry_response.content


        if isinstance(answer, list):

            answer = "\n".join(
                str(item)
                for item in answer
            )


        answer = str(answer).strip()


        # -------------------------------------------------
        # Retry debugging
        # -------------------------------------------------

        print("\n===== RETRY RESPONSE LENGTH =====")
        print(len(answer))

        print("\n===== RAW RETRY RESPONSE =====")
        print(repr(answer))


    # =====================================================
    # FINAL EMPTY RESPONSE CHECK
    # =====================================================

    if not answer:

        return (
            "The study planner could not generate a plan. "
            "Please try again.",
            documents
        )


    # =====================================================
    # RETURN PLAN
    # =====================================================

    return answer, documents


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    subject = "DBMS"

    duration_days = 7

    hours_per_day = 2


    # -----------------------------------------------------
    # Generate study plan
    # -----------------------------------------------------

    plan, documents = create_study_plan(
        subject,
        duration_days,
        hours_per_day
    )


    # -----------------------------------------------------
    # Display generated plan
    # -----------------------------------------------------

    print(
        "\n===== SYLLABUS-AWARE STUDY PLAN =====\n"
    )

    print(plan)


    # -----------------------------------------------------
    # Display sources
    # -----------------------------------------------------

    print("\n===== SOURCES =====\n")


    for document in documents:

        print(
            f"- {document.metadata.get('source')} "
            f"(page {document.metadata.get('page')})"
        )