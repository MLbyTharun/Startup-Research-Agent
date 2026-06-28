# graph.py
# LangGraph as the outer orchestrator wrapping the CrewAI crew
#
# Why LangGraph on top of CrewAI?
# - Manages overall state across the research flow
# - Handles retries automatically if the crew fails
# - Clean separation: CrewAI handles WHO does WHAT, LangGraph handles FLOW

from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from crew import run_research
import time


# STATE 

class ResearchState(TypedDict):
    company: str
    brief: Optional[str]
    token_usage: Optional[dict]
    error: Optional[str]
    retry_count: int


# NODES
def validate_input(state: ResearchState) -> ResearchState:
    """Validate company name before spending API credits"""
    company = state["company"].strip()

    if not company:
        return {**state, "error": "Company name cannot be empty"}
    if len(company) < 2:
        return {**state, "error": "Company name too short"}

    return {**state, "company": company, "error": None}


def run_crew(state: ResearchState) -> ResearchState:
    """Run the CrewAI research crew — all the agent work happens here"""
    try:
        print(f"\n🚀 Starting research crew for: {state['company']}")
        result = run_research(state["company"])

        return {
            **state,
            "brief": result["brief"],
            "token_usage": result["token_usage"],
            "error": None,
        }

    except Exception as e:
        print(f"❌ Crew failed: {e}")
        return {
            **state,
            "error": str(e),
            "retry_count": state.get("retry_count", 0) + 1,
        }


def format_output(state: ResearchState) -> ResearchState:
    from datetime import datetime

    brief = state.get("brief") or ""

    # Check if agent flagged company as not found
    if brief.startswith("COMPANY_NOT_FOUND:"):
        return {**state, "error": brief.replace("COMPANY_NOT_FOUND: ", "")}

    timestamp = datetime.now().strftime("%B %d, %Y")
    header = f"*Generated: {timestamp}*\n\n---\n\n"
    return {**state, "brief": header + brief}


# CONDITIONAL EDGES 

def should_retry(state: ResearchState) -> str:
    """Retry up to 2 times on failure, then give up"""
    if state.get("error") and state.get("retry_count", 0) < 2:
        print(f"🔄 Retrying... (attempt {state['retry_count']})")
        time.sleep(2)
        return "retry"
    return "continue"


# BUILD GRAPH 

def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("validate", validate_input)
    graph.add_node("research", run_crew)
    graph.add_node("format", format_output)

    graph.set_entry_point("validate")

    graph.add_conditional_edges(
        "validate",
        lambda s: "error" if s.get("error") else "continue",
        {"continue": "research", "error": END}
    )

    graph.add_conditional_edges(
        "research",
        should_retry,
        {"retry": "research", "continue": "format"}
    )

    graph.add_edge("format", END)

    return graph.compile()


app = build_graph()


def research_company(company: str) -> dict:
    """Top-level function called by the Streamlit UI"""
    initial_state: ResearchState = {
        "company": company,
        "brief": None,
        "token_usage": None,
        "error": None,
        "retry_count": 0,
    }
    return app.invoke(initial_state)

