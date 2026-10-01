from langgraph.graph import StateGraph, START, END

from agents.state import AgentState
from agents.router import route_query
from agents.academic_agent import academic_agent
from agents.study_agent import study_agent
from agents.general_agent import general_agent


def route_after_classifier(state):

    intent = state["intent"]

    if intent == "ACADEMIC":
        return "academic"

    elif intent == "STUDY_PLAN":
        return "study"

    else:
        return "general"


def build_graph():

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("router", route_query)
    graph.add_node("academic", academic_agent)
    graph.add_node("study", study_agent)
    graph.add_node("general", general_agent)

    # START → Router
    graph.add_edge(START, "router")

    # Router → appropriate agent
    graph.add_conditional_edges(
        "router",
        route_after_classifier,
        {
            "academic": "academic",
            "study": "study",
            "general": "general"
        }
    )

    # Agents → END
    graph.add_edge("academic", END)
    graph.add_edge("study", END)
    graph.add_edge("general", END)

    return graph.compile()