from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


prompt = """
Create a simple 7-day study plan for DBMS.

Topics:
1. Relational Model
2. Relational Algebra
3. SQL
4. Functional Dependencies
5. Normalization
6. Storage and Indexing
7. Transactions and Concurrency Control

Return the answer as a markdown table.
"""


print("===== TESTING GROQ =====")

response = llm.invoke(prompt)

print("\nResponse object:")
print(response)

print("\nResponse content:")
print(repr(response.content))