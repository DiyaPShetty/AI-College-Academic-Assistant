from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


NO_INFO_MESSAGE = (
    "I couldn't find this information in the available "
    "college documents."
)


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


def generate_academic_answer(
    question,
    documents,
    conversation_history=None
):

    if not documents:
        return NO_INFO_MESSAGE

    context_parts = []

    for i, document in enumerate(documents, start=1):

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {i}
Document: {source}
Page: {page}

Content:
{document.page_content}
"""
        )

    context = "\n".join(context_parts)

    history_text = ""

    if conversation_history:
        history_text = "\n".join(
            f"{m.get('role')}: {m.get('content')}"
            for m in conversation_history[-6:]
        )

    prompt = f"""
You are the academic assistant for NMAM Institute of Technology.

Answer the student's question ONLY using the official
college document context below.

CONVERSATION:
{history_text}

QUESTION:
{question}

OFFICIAL DOCUMENT CONTEXT:
{context}

STRICT RULES:

1. Do not invent college rules.
2. Do not invent dates, marks, credits or requirements.
3. Do not use outside knowledge for college-specific facts.
4. If the context does not support the answer, say exactly:

"I couldn't find this information in the available college documents."

5. Cite supporting sources using:
[Source: filename, Page: X]

Give a clear answer.
"""

    response = get_llm().invoke(prompt)

    return str(response.content).strip()


def ask_academic_assistant(question):

    # Backward-compatible helper for old tests.
    from rag.retriever import get_retriever

    retriever = get_retriever()

    documents = retriever.invoke(question)

    answer = generate_academic_answer(
        question,
        documents
    )

    return answer, documents
