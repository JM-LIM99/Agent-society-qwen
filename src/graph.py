"""
Graph assembly.

Topology:

    START
      |
      v
  read_background -> read_methodology -> read_experiments -> read_limitations
      |
      v
   critic ----(issues & under limit)----> reviser --+
      |                                              |
      | (approved or limit hit)                      |
      |   ^------------------------------------------+
      v
  synthesizer -> architect -> reviewer ----(revise & under limit)--+
                                  |                                 |
                                  | (approved or limit hit)         |
                                  v          ^----------------------+
                                 END   (back to architect)

The two conditional edges are where "negotiation" lives:
  - Critic can bounce reading back to the Reviser.
  - Reviewer can bounce the design back to the Architect.
Both are bounded by revision counters so the graph always terminates.
"""
from langgraph.graph import StateGraph, START, END
from .agents import read_section
from .state import AgentState, MAX_REVISIONS, MAX_DESIGN_REVISIONS
from .config import READER_ROLES
from .agents import (
    critic, reviser, synthesizer,
    architect, explainer, reviewer, coder,
)


def route_after_critic(state: AgentState) -> str:
    if "APPROVED" in state["critique"]:
        return "synthesizer"
    if state.get("revisions", 0) >= MAX_REVISIONS:
        return "synthesizer"
    
    return "reviser"

def route_after_reviewer(state: AgentState) -> str:
    if "APPROVED" in state["review"]:
        return "coder"
    if state.get("design_revisions", 0) >= MAX_DESIGN_REVISIONS:
        return "coder" 
    return "architect"

def build_graph():
    g = StateGraph(AgentState)

    for role in READER_ROLES:
        g.add_node(f"read_{role}", lambda s, r= role: read_section(s,r))
    g.add_node("critic", critic)
    g.add_node("reviser", reviser)
    g.add_node("synthesizer", synthesizer)
    g.add_node("architect", architect)
    g.add_node("explainer", explainer)
    g.add_node("reviewer", reviewer)
    g.add_node("coder", coder)

    g.set_entry_point("read_background")
    g.add_edge("read_background", "read_methodology")
    g.add_edge("read_methodology", "read_experiments")
    g.add_edge("read_experiments", "read_limitations")
    g.add_edge("read_limitations", "critic")

    g.add_conditional_edges("critic", route_after_critic,
                            {"reviser": "reviser", "synthesizer": "synthesizer"})
    g.add_edge("reviser", "critic")
    g.add_edge("synthesizer", "explainer")
    g.add_edge("explainer", "architect")
    g.add_conditional_edges("reviewer", route_after_reviewer,
                            {"architect": "architect", "coder": "coder"})
    g.add_edge("architect", "reviewer")
    g.add_edge("coder", END)

    return g.compile()
