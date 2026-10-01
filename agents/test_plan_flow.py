from agents.graph import build_graph


graph = build_graph()


state = {
    "user_query": "Create a 7 day DBMS study plan",
    "intent": "",
    "retrieved_docs": [],
    "answer": "",
    "study_plan": "",
    "modification": ""
}


print("\n==============================")
print("CREATING STUDY PLAN")
print("==============================")

result = graph.invoke(state)

print("\nINTENT:")
print(result["intent"])

print("\nSTUDY PLAN:")
print(result["study_plan"])


# Now modify the same plan
result["user_query"] = "Move the Day 5 topics to Day 6."

print("\n==============================")
print("MODIFYING STUDY PLAN")
print("==============================")

result = graph.invoke(result)

print("\nINTENT:")
print(result["intent"])

print("\nUPDATED STUDY PLAN:")
print(result["study_plan"])