from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


def create_study_plan(subject, duration_days, hours_per_day):
    llm = get_llm()

    prompt = f"""
You are a college study planner.

Create a practical study plan for the following:

Subject: {subject}
Number of days: {duration_days}
Study hours per day: {hours_per_day}

Divide the study workload across the days.

For each day provide:
- Day number
- Topics to study
- Study duration
- A short task or goal

Do not invent specific syllabus topics if they are not provided.
If topics are not provided, create a general study structure and
clearly indicate that the topics should be mapped to the student's syllabus.

Return the plan in a clear format.
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":
    subject = "DBMS"
    duration_days = 7
    hours_per_day = 2

    plan = create_study_plan(
        subject,
        duration_days,
        hours_per_day
    )

    print("\n===== STUDY PLAN =====\n")
    print(plan)