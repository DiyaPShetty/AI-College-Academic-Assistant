from agents.graph import build_graph


graph = build_graph()


queries = [
    "What is the minimum attendance required?",
    "Create a 7 day DBMS study plan",
    "What is Python?"
]


for query in queries:

    print("\n" + "=" * 60)
    print("QUESTION:", query)
    print("=" * 60)

    initial_state = {
        "user_query": query,
        "intent": "",
        "retrieved_docs": [],
        "answer": "",
        "study_plan": "",
        "modification": ""
    }

    result = graph.invoke(initial_state)

    print("\nINTENT:", result["intent"])
    print("\nANSWER:\n")
    print(result["answer"])
