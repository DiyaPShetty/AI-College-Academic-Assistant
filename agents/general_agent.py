from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def general_agent(state):

    query = state["user_query"]

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = f"""
You are a helpful college AI assistant.

Answer the student's question clearly and concisely.

Student question:
{query}
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "answer": response.content
    }
