"""
Shared state passed between all agents in the graph.

This is the "blackboard" of the agent society: every agent reads from
and writes to this single typed dict. Conflict resolution works by the
Critic writing `issues`, which routes the graph back to the Reviser.
"""
from typing import TypedDict, List, Dict


class AgentState(TypedDict):
    # --- Reading phase ---
    analyses: Dict[str, str]       # section name -> analysis text
    issues: List[dict]             # Critic-flagged problems (empty = approved)
    revision_count: int            # guards the Critic<->Reviser loop

    # --- Design phase ---
    synthesis: str                 # core contributions extracted
    design: str                    # proposed system design
    review_feedback: str           # Reviewer's notes
    design_revision_count: int
    code: str    # generated code components for the design.



# The four fixed reader roles and the query each uses to pull context.
SECTIONS = {
    "background":   "introduction motivation problem statement background prior work",
    "methodology":  "proposed model architecture method approach how it works",
    "experiments":  "experiments results evaluation benchmark metrics performance",
    "limitations":  "limitations weaknesses constraints future work assumptions",
}

# Loop guards: how many times a phase may be sent back for rework.
MAX_REVISIONS = 2
MAX_DESIGN_REVISIONS = 2
