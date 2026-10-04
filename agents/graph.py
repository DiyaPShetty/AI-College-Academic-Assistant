from typing import Literal

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agents.state import AgentState
from agents.router import route_query

from agents.workflow_nodes import (
    retrieval_node,
    academic_generation_node,
    study_plan_node,
    plan_validator,
    general_generation_node,
    calculator_node,
    response_review_node,
)


# =========================================================
# ROUTING AFTER QUESTION ANALYSIS
# =========================================================

def route_after_analysis(
    state
) -> Literal[
    "academic",
    "study",
    "general",
    "tool"
]:

    intent = state.get("intent", "GENERAL")

    if intent == "ACADEMIC":
        return "academic"

    if intent == "STUDY_PLAN":
        return "study"

    if intent == "TOOL":
        return "tool"

    return "general"


# =========================================================
# ROUTING AFTER RETRIEVAL
# =========================================================

def route_after_retrieval(state):

    if state.get("retrieval_relevant", False):
        return "generate"

    return "generate"


# =========================================================
# BUILD GRAPH
# =========================================================

def build_graph():

    graph = StateGraph(AgentState)

    # =====================================================
    # NODES
    # =====================================================

    graph.add_node(
        "question_analysis",
        route_query
    )

    graph.add_node(
        "information_retrieval",
        retrieval_node
    )

    graph.add_node(
        "response_generation",
        academic_generation_node
    )

    graph.add_node(
        "study_planner",
        study_plan_node
    )

    graph.add_node(
        "plan_validator",
        plan_validator
    )

    graph.add_node(
        "general_response",
        general_generation_node
    )

    graph.add_node(
        "calculator_tool",
        calculator_node
    )

    graph.add_node(
        "response_review",
        response_review_node
    )

    # =====================================================
    # START
    # =====================================================

    graph.add_edge(
        START,
        "question_analysis"
    )

    # =====================================================
    # QUESTION ANALYSIS → ROUTE
    # =====================================================

    graph.add_conditional_edges(
        "question_analysis",
        route_after_analysis,
        {
            "academic": "information_retrieval",
            "study": "study_planner",
            "general": "general_response",
            "tool": "calculator_tool"
        }
    )

    # =====================================================
    # ACADEMIC
    # =====================================================

    graph.add_edge(
        "information_retrieval",
        "response_generation"
    )

    graph.add_edge(
        "response_generation",
        "response_review"
    )

    # =====================================================
    # STUDY PLAN
    # =====================================================

    graph.add_edge(
        "study_planner",
        "plan_validator"
    )

    graph.add_edge(
        "plan_validator",
        "response_review"
    )

    # =====================================================
    # GENERAL
    # =====================================================

    graph.add_edge(
        "general_response",
        "response_review"
    )

    # =====================================================
    # TOOL
    # =====================================================

    graph.add_edge(
        "calculator_tool",
        "response_review"
    )

    # =====================================================
    # FINAL
    # =====================================================

    graph.add_edge(
        "response_review",
        END
    )

    return graph.compile()
