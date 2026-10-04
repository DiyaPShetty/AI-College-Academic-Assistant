import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import streamlit as st

from agents.app_agent import run_agent


st.set_page_config(
    page_title="AI College Academic Assistant",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: #f0f2f6;
    }
    .stChatMessage[data-testid="stChatMessage"][data-type="user"] {
        background-color: #e3f2fd;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "study_plan" not in st.session_state:
    st.session_state.study_plan = ""


# =========================================================
# HEADER
# =========================================================

st.title("🎓 AI College Academic Assistant")

st.markdown(
    """
    Ask questions about college academics, regulations, syllabus,
    or create and modify personalized study plans.
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📊 System Status")

    st.success("✅ RAG enabled")
    st.success("✅ LangGraph workflow enabled")
    st.success("✅ Study planner enabled")
    st.success("✅ Calculator tool enabled")

    st.divider()

    st.header("💡 Quick Tips")

    st.info("""
    **Academic Questions:**
    - "What is the minimum attendance requirement?"
    - "How many credits are required for graduation?"

    **Study Plans:**
    - "Create a 7-day study plan for DBMS with 2 hours per day"
    - "Move Day 5 topics to Day 6"
    """)

    st.divider()

    if st.button("🗑️ Clear conversation"):

        st.session_state.messages = []
        st.session_state.study_plan = ""

        st.rerun()


# =========================================================
# PREVIOUS MESSAGES
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

user_query = st.chat_input(
    "Ask your academic question..."
)


if user_query:

    # -----------------------------------------------------
    # SHOW USER
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)


    # -----------------------------------------------------
    # PREVIOUS HISTORY
    # -----------------------------------------------------

    history = st.session_state.messages[:-1]


    # -----------------------------------------------------
    # RUN LANGGRAPH
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Analyzing → retrieving → generating → reviewing..."
        ):

            try:
                result = run_agent(
                    user_query=user_query,
                    current_plan=st.session_state.study_plan,
                    conversation_history=history
                )
            except Exception as e:
                st.error(
                    f"An error occurred while processing your request: {str(e)}"
                )
                st.info(
                    "Please check your environment variables and try again."
                )
                st.stop()


        answer = str(
            result.get(
                "answer",
                "I couldn't generate a response."
            )
        )

        st.markdown(answer)


        # -------------------------------------------------
        # METADATA
        # -------------------------------------------------

        intent = result.get(
            "intent",
            "UNKNOWN"
        )

        st.caption(
            f"🎯 Intent: {intent}"
        )


        # -------------------------------------------------
        # SOURCES
        # -------------------------------------------------

        documents = result.get(
            "retrieved_docs",
            []
        )

        if documents:

            with st.expander(
                "📚 Sources"
            ):

                seen = set()

                for document in documents:

                    source = document.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = document.metadata.get(
                        "page",
                        "Unknown"
                    )

                    key = (
                        source,
                        page
                    )

                    if key in seen:
                        continue

                    seen.add(key)

                    st.write(
                        f"• {source} — page {page}"
                    )


        # -------------------------------------------------
        # REVIEW STATUS
        # -------------------------------------------------

        review_status = result.get(
            "review_status"
        )

        if review_status:

            if review_status == "PASS":
                st.success(
                    f"✅ Response review: {review_status}"
                )
            elif review_status == "FAILED":
                st.warning(
                    f"⚠️ Response review: {review_status}"
                )
            else:
                st.caption(
                    f"Response review: {review_status}"
                )


    # -----------------------------------------------------
    # SAVE ASSISTANT MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # -----------------------------------------------------
    # SAVE STUDY PLAN
    # -----------------------------------------------------

    if result.get("study_plan"):

        if result.get("intent") == "STUDY_PLAN":

            st.session_state.study_plan = (
                result["study_plan"]
            )
