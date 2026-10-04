"""
Configuration module for the AI College Academic Assistant.

This module centralizes configuration constants and provides
utilities for loading and validating environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =========================================================
# LLM CONFIGURATION
# =========================================================

GROQ_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 4096

# =========================================================
# EMBEDDINGS CONFIGURATION
# =========================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================================================
# VECTOR DATABASE CONFIGURATION
# =========================================================

CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "college_knowledge"

# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

RETRIEVAL_K = 5
RETRIEVAL_THRESHOLD = 0.30

# =========================================================
# CONVERSATION CONFIGURATION
# =========================================================

CONVERSATION_HISTORY_LIMIT = 6

# =========================================================
# ENVIRONMENT VARIABLE VALIDATION
# =========================================================


def get_env_var(var_name: str, default: str = None, required: bool = False) -> str:
    """
    Get an environment variable with optional validation.

    Args:
        var_name (str): Name of the environment variable.
        default (str, optional): Default value if variable is not set.
        required (bool): If True, raises an error when variable is missing.

    Returns:
        str: The environment variable value.

    Raises:
        ValueError: If the variable is required but not set.
    """
    value = os.getenv(var_name, default)

    if required and not value:
        raise ValueError(
            f"Required environment variable '{var_name}' is not set. "
            f"Please add it to your .env file."
        )

    return value


def validate_environment():
    """
    Validate that all required environment variables are set.

    Returns:
        bool: True if all required variables are set.

    Raises:
        ValueError: If any required environment variable is missing.
    """
    required_vars = ["GROQ_API_KEY"]

    missing_vars = [
        var for var in required_vars
        if not os.getenv(var)
    ]

    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}. "
            f"Please add them to your .env file. "
            f"See .env.example for reference."
        )

    return True


# =========================================================
# API KEYS
# =========================================================

GROQ_API_KEY = get_env_var("GROQ_API_KEY", required=True)
