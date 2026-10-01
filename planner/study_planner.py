from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=4096
    )


# ---------------------------------------------------------
# Vector Store
# ---------------------------------------------------------

def get_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    return vectorstore


# ---------------------------------------------------------
# Retrieve DBMS syllabus
# ---------------------------------------------------------

def get_syllabus_context(subject):

    vectorstore = get_vectorstore()

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

    else:

        relevant_documents = vectorstore.similarity_search(
            f"{subject} official syllabus",
            k=5
        )


    # Remove duplicates
    unique_documents = []

    seen = set()

    for document in relevant_documents:

        content = document.page_content.strip()

        if content not in seen:

            seen.add(content)
            unique_documents.append(document)


    # Sort by page
    unique_documents.sort(
        key=lambda document: document.metadata.get("page", 0)
    )


    # Keep only a small amount of context
    documents = unique_documents[:3]


    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    return context, documents


# ---------------------------------------------------------
# Create Study Plan
# ---------------------------------------------------------

def create_study_plan(
    subject,
    duration_days,
    hours_per_day
):

    syllabus_context, documents = get_syllabus_context(subject)

    print("\n===== RETRIEVED SYLLABUS CONTEXT =====\n")
    print(syllabus_context)

    print("\n===== END CONTEXT =====\n")


    if not syllabus_context.strip():

        return (
            "No relevant syllabus information was retrieved.",
            documents
        )


    llm = get_llm()


    # -----------------------------------------------------
    # Short, simple prompt
    # -----------------------------------------------------

    prompt = f"""
Create a {duration_days}-day study plan for {subject}.

Study time: {hours_per_day} hours per day.

Use ONLY the official syllabus below.

SYLLABUS:
{syllabus_context}

Rules:
- Use only topics explicitly present in the syllabus.
- Do not invent topics.
- Do not use outside knowledge.
- Do not add resources or textbooks.
- Do not add new units.
- Do not add subtopics that are not written in the syllabus.

Return ONLY a markdown table.

Columns:
Day | Unit | Topics | Duration | Study Goal

Distribute the syllabus topics across {duration_days} days.
"""


    print("\n===== CALLING GROQ =====\n")


    response = llm.invoke(prompt)


    print("\n===== RESPONSE OBJECT =====")
    print(response)


    answer = response.content


    if isinstance(answer, list):

        answer = "\n".join(
            str(item)
            for item in answer
        )


    answer = str(answer).strip()


    print("\n===== LLM RESPONSE LENGTH =====")
    print(len(answer))


    print("\n===== RAW LLM RESPONSE =====")
    print(repr(answer))


    if not answer:

        return (
            "The LLM returned an empty response.",
            documents
        )


    return answer, documents


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    subject = "DBMS"
    duration_days = 7
    hours_per_day = 2

    plan, documents = create_study_plan(
        subject,
        duration_days,
        hours_per_day
    )


    print("\n===== SYLLABUS-AWARE STUDY PLAN =====\n")

    print(plan)


    print("\n===== SOURCES =====\n")

    for document in documents:

        print(
            f"- {document.metadata.get('source')} "
            f"(page {document.metadata.get('page')})"
        )