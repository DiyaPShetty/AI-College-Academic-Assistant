from dotenv import load_dotenv
from langchain_groq import ChatGroq

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
# MODIFY STUDY PLAN
# =========================================================

def modify_study_plan(current_plan, modification):

    llm = get_llm()


    # =====================================================
    # MAIN PROMPT
    # =====================================================

    prompt = f"""
Modify the existing study plan according to the
student's request.

==================================================
CURRENT STUDY PLAN
==================================================

{current_plan}

==================================================
STUDENT REQUEST
==================================================

{modification}

==================================================
STRICT RULES
==================================================

1. The CURRENT STUDY PLAN is the source of truth.

2. Preserve every existing topic unless the student
   explicitly asks to remove or change it.

3. Do NOT invent new topics.

4. Do NOT add topics that are not already present
   in the current study plan.

5. Do NOT use outside knowledge.

6. Do NOT add textbooks.

7. Do NOT add study resources.

8. Do NOT add chapter numbers.

9. Do NOT add page numbers.

10. Do NOT add textbook references.

11. Do NOT add reference codes such as:
    T1: 6.1
    T2: 8.2
    Chapter 8
    Page 120

12. Do NOT create new units.

13. Do NOT rename existing units.

14. Preserve the existing Unit names exactly.

15. Preserve the existing topic names exactly
    whenever possible.

16. Keep the same number of days unless the student
    explicitly requests a different number.

17. Keep the duration of each day unchanged unless
    the student explicitly requests a duration change.

18. If moving topics creates a conflict between two
    days, rearrange the affected days so that:
    - no topic is lost
    - no topic is duplicated
    - the requested modification is satisfied
    - the daily duration remains unchanged

19. Keep each Study Goal associated with its topics.

20. Do not add explanations before or after the table.

==================================================
OUTPUT FORMAT
==================================================

Return ONLY the complete updated markdown table.

Use exactly these columns:

| Day | Unit | Topics | Duration | Study Goal |
|-----|------|--------|----------|------------|

Do not return anything except the table.
"""


    # =====================================================
    # CALL GROQ
    # =====================================================

    print("\n===== CALLING GROQ FOR MODIFICATION =====\n")

    response = llm.invoke(prompt)


    # =====================================================
    # RESPONSE OBJECT
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

    print("\n===== RESPONSE LENGTH =====")
    print(len(answer))

    print("\n===== RAW RESPONSE =====")
    print(repr(answer))


    # =====================================================
    # RETRY IF EMPTY
    # =====================================================

    if not answer:

        print(
            "\n===== EMPTY RESPONSE - "
            "RETRYING MODIFICATION =====\n"
        )


        retry_prompt = f"""
Modify this existing study plan.

CURRENT PLAN:

{current_plan}

STUDENT REQUEST:

{modification}

Rules:

- The current plan is the source of truth.
- Preserve all existing topics.
- Do not invent topics.
- Do not add new topics.
- Do not add new units.
- Do not rename units.
- Preserve existing topic names.
- Do not add textbooks.
- Do not add resources.
- Do not add chapter numbers.
- Do not add page numbers.
- Do not add reference codes.
- Keep the same number of days.
- Keep the existing duration for each day.
- If two days conflict, rearrange the affected days.
- Do not lose or duplicate any topics.
- Apply the student's requested modification.
- Return ONLY the complete markdown table.

Format:

| Day | Unit | Topics | Duration | Study Goal |
|-----|------|--------|----------|------------|
"""


        retry_response = llm.invoke(
            retry_prompt
        )


        print("\n===== RETRY RESPONSE =====")
        print(retry_response)


        answer = retry_response.content


        if isinstance(answer, list):

            answer = "\n".join(
                str(item)
                for item in answer
            )


        answer = str(answer).strip()


        print("\n===== RETRY RESPONSE LENGTH =====")
        print(len(answer))

        print("\n===== RAW RETRY RESPONSE =====")
        print(repr(answer))


    # =====================================================
    # FINAL CHECK
    # =====================================================

    if not answer:

        return (
            "The study plan could not be modified. "
            "Please try the modification again.",
            []
        )


    # =====================================================
    # RETURN UPDATED PLAN
    # =====================================================

    return answer, []


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    current_plan = """
| Day | Unit | Topics | Duration | Study Goal |
|-----|------|--------|----------|------------|
| 1 | Basic SQL | SQL Data Definition and Data Types | 2 hours | Cover SQL Data Definition and Data Types |
| 2 | Basic SQL | Views and Schema Modification | 2 hours | Cover Views and Schema Modification |
| 3 | Basics of Functional Dependencies and Normalization for Relational Databases | Functional Dependencies, Normal Forms Based on Primary Keys | 2 hours | Cover Functional Dependencies and Normal Forms |
| 4 | Relational Database Design Algorithms and Further Dependencies | Inference Rules, Equivalence, Minimal cover | 2 hours | Cover Inference Rules, Equivalence and Minimal cover |
| 5 | UNIT-III Storage and Indexing, Query Evaluation, Transaction Management | File Organizations and Indexing, Index Data structures, Comparison of File Organizations | 2 hours | Cover File Organizations and Indexing |
| 6 | UNIT-III Storage and Indexing, Query Evaluation, Transaction Management | B+ Tree: A Dynamic Index Structure, Introduction to Query Optimization | 2 hours | Cover B+ Tree and Introduction to Query Optimization |
| 7 | UNIT-III Storage and Indexing, Query Evaluation, Transaction Management | The ACID Properties, Transactions and Schedules, Concurrent Execution of Transactions | 2 hours | Cover transaction management topics |
"""


    modification = (
        "Move the Day 5 topics to Day 6."
    )


    # =====================================================
    # RUN MODIFICATION
    # =====================================================

    plan, documents = modify_study_plan(
        current_plan,
        modification
    )


    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    print(
        "\n===== UPDATED PLAN =====\n"
    )

    print(plan)