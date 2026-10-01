import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag.retriever import get_retriever


load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


def ask_academic_assistant(question):
    retriever = get_retriever()

    # Retrieve relevant college information
    documents = retriever.invoke(question)

    # Combine retrieved chunks
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are an AI academic assistant for NMAM Institute of Technology.

Answer the student's question using ONLY the information provided
in the college documents below.

If the answer cannot be found in the provided documents, say:
"I couldn't find this information in the available college documents."

Do not invent college rules, regulations, dates, marks, or requirements.

College document context:
-------------------------
{context}
-------------------------

Student question:
{question}

Give a clear and concise answer.
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content, documents


if __name__ == "__main__":
    question = "What is the minimum attendance requirement for students?"

    answer, documents = ask_academic_assistant(question)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")

    for document in documents:
        print(
            f"- {document.metadata.get('source')} "
            f"(page {document.metadata.get('page')})"
        )