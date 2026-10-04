from agents.app_agent import run_agent


print("\n==============================")
print("CREATING STUDY PLAN")
print("==============================")


result = run_agent(
    "Create a 7 day DBMS study plan. "
    "I can study 2 hours per day."
)


print("\nINTENT:")
print(result["intent"])

print("\nSUBJECTS:")
print(result["subjects"])

print("\nDAYS:")
print(result["duration_days"])

print("\nHOURS:")
print(result["hours_per_day"])

print("\nSTUDY PLAN:")
print(result["study_plan"])


# =========================================================
# MODIFY THE SAME PLAN
# =========================================================

print("\n==============================")
print("MODIFYING STUDY PLAN")
print("==============================")


result["user_query"] = (
    "Move the Day 5 topics to Day 6."
)

result = run_agent(
    user_query=result["user_query"],
    current_plan=result["study_plan"],
    conversation_history=[]
)


print("\nINTENT:")
print(result["intent"])

print("\nUPDATED STUDY PLAN:")
print(result["study_plan"])
