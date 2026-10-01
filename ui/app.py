import sys
import os


# =========================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =========================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


import streamlit as st
from agents.app_agent import run_agent


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI College Academic Assistant",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

# Store conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Store the current study plan
if "study_plan" not in st.session_state:
    st.session_state.study_plan = ""


# =========================================================
# TITLE
# =========================================================

st.title("🎓 AI College Academic Assistant")

st.write(
    "Ask questions about academics, syllabus, college regulations, "
    "or request and modify a study plan."
)


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        content = message["content"]

        # Clean HTML line-break tags returned by the LLM
        content = content.replace("<br>", " • ")
        content = content.replace("<br/>", " • ")
        content = content.replace("<br />", " • ")

        st.markdown(content)


# =========================================================
# CHAT INPUT
# =========================================================

user_query = st.chat_input(
    "Ask your question..."
)


# =========================================================
# PROCESS USER QUERY
# =========================================================

if user_query:

    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_query)


    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )


    # -----------------------------------------------------
    # CALL LANGGRAPH
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            result = run_agent(
                user_query,
                st.session_state.study_plan
            )


        # -------------------------------------------------
        # GET RESULT
        # -------------------------------------------------

        answer = result["answer"]
        intent = result["intent"]


        # -------------------------------------------------
        # MAKE SURE ANSWER IS A STRING
        # -------------------------------------------------

        if answer is None:

            answer = "I couldn't generate a response."

        else:

            answer = str(answer)


        # -------------------------------------------------
        # SAVE / UPDATE STUDY PLAN
        # -------------------------------------------------

        if intent == "STUDY_PLAN":

            if answer.strip():

                st.session_state.study_plan = answer


        # -------------------------------------------------
        # CLEAN LLM HTML TAGS
        # -------------------------------------------------

        display_answer = answer

        display_answer = display_answer.replace(
            "<br>",
            " • "
        )

        display_answer = display_answer.replace(
            "<br/>",
            " • "
        )

        display_answer = display_answer.replace(
            "<br />",
            " • "
        )


        # -------------------------------------------------
        # DISPLAY ANSWER
        # -------------------------------------------------

        st.markdown(display_answer)


        # -------------------------------------------------
        # DISPLAY INTENT
        # -------------------------------------------------

        st.caption(
            f"Intent detected: {intent}"
        )


    # -----------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )