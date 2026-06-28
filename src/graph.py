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

from .state import AgentState, SECTIONS, MAX_REVISIONS, MAX_DESIGN_REVISIONS
from . import agents


def _route_after_critic(state):
    if state["issues"] and state["revision_count"] <= MAX_REVISIONS:
        return "reviser"
    return "synthesizer"


def _route_after_reviewer(state):
    if state["review_feedback"] and state["design_revision_count"] <= MAX_DESIGN_REVISIONS:
        return "architect"
    return "coder"


def build_graph(retriever, llm):
    g = StateGraph(AgentState)

    # reader nodes (fixed four)
    for section in SECTIONS:
        g.add_node(f"read_{section}", agents.make_reader(section, retriever, llm))

    g.add_node("critic", agents.make_critic(llm))
    g.add_node("reviser", agents.make_reviser(retriever, llm))
    g.add_node("synthesizer", agents.make_synthesizer(llm))
    g.add_node("architect", agents.make_architect(llm))
    g.add_node("reviewer", agents.make_reviewer(llm))
    g.add_node("coder", agents.make_coder(llm))

    # reading chain (sequential, deterministic order)
    sections = list(SECTIONS.keys())
    g.add_edge(START, f"read_{sections[0]}")
    for a, b in zip(sections, sections[1:]):
        g.add_edge(f"read_{a}", f"read_{b}")
    g.add_edge(f"read_{sections[-1]}", "critic")

    # critic <-> reviser loop
    g.add_conditional_edges("critic", _route_after_critic,
                            {"reviser": "reviser", "synthesizer": "synthesizer"})
    g.add_edge("reviser", "critic")

    # design phase
    g.add_edge("synthesizer", "architect")
    g.add_edge("architect", "reviewer")
    g.add_conditional_edges("reviewer", _route_after_reviewer,
                            {"architect": "architect", "coder": "coder"})
    g.add_edge("coder", END)

    return g.compile()
