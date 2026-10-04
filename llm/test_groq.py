"""
Test script for Groq LLM connection.

This script verifies that the Groq API key is correctly configured
and that the LLM can generate responses.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load variables from .env
load_dotenv()

# Create Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# Test question
response = llm.invoke(
    "What is a database? Explain in one simple sentence."
)

print("\nGroq Response:")
print(response.content)
