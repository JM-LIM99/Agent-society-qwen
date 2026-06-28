"""
Agent definitions.

Each function here is (or builds) a LangGraph node. A node takes the
shared AgentState, does its job (usually one LLM call grounded in
retrieved chunks), and returns a partial state update.

Roles
-----
Reader      x4  : analyze one section of the paper (background / method /
                  experiments / limitations), grounded in retrieved chunks.
Critic          : cross-check the four analyses for contradictions,
                  unsupported claims, or gaps. Emits `issues`.
Reviser         : rewrite only the flagged sections using Critic feedback.
Synthesizer     : merge the four analyses into core contributions.
Architect       : turn contributions into a concrete system design.
Reviewer        : judge the design; send back if it's weak.
"""
import json
from langchain_openai import ChatOpenAI

from . import config
from .state import SECTIONS


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_llm():
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.QWEN_API_KEY,
        base_url=config.QWEN_BASE_URL
    )


def _retrieve(retriever, query: str) -> str:
    """Pull relevant chunks and flatten them into a context string."""
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON; tolerate model sloppiness."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # last resort: grab the outermost braces
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            return json.loads(cleaned[start:end + 1])
        raise


# --------------------------------------------------------------------------
# Reader nodes (one factory, four instances)
# --------------------------------------------------------------------------
def make_reader(section: str, retriever, llm):
    """Build a reader node bound to one section."""
    query = SECTIONS[section]

    def reader(state):
        context = _retrieve(retriever, query)

        # If this section was previously flagged, fold the feedback in.
        prior_issue = next(
            (i for i in state.get("issues", []) if i["section"] == section),
            None,
        )
        feedback = (
            f"\nA reviewer flagged your previous analysis: {prior_issue['problem']}\n"
            f"Address this specifically."
            if prior_issue else ""
        )

        prompt = (
            f"You are the **{section}** analyst on a paper-analysis team.\n"
            f"Using ONLY the excerpts below, write a tight, factual analysis "
            f"of the paper's {section}. Cite specifics (numbers, names, "
            f"mechanisms). Do not invent anything not in the text.{feedback}\n\n"
            f"=== EXCERPTS ===\n{context}\n=== END ===\n\n"
            f"Analysis of {section}:"
        )
        result = llm.invoke(prompt).content
        # merge into the analyses dict without clobbering siblings
        analyses = dict(state.get("analyses", {}))
        analyses[section] = result
        return {"analyses": analyses}

    return reader


# --------------------------------------------------------------------------
# Critic node  (conflict / consistency check)
# --------------------------------------------------------------------------
def make_critic(llm):
    def critic(state):
        analyses = state["analyses"]
        joined = "\n\n".join(f"## {k}\n{v}" for k, v in analyses.items())

        prompt = (
            "You are the Critic on a paper-analysis team. Below are four "
            "section analyses written by different agents. Find genuine "
            "problems: factual contradictions between sections, claims with "
            "no support, or important gaps.\n\n"
            "Respond ONLY with JSON in this exact shape:\n"
            '{"verdict": "approve" | "revise", '
            '"issues": [{"section": "<background|methodology|experiments|limitations>", '
            '"problem": "<what is wrong and why>"}]}\n'
            "If everything is consistent and well-supported, return verdict "
            '"approve" with an empty issues list. Only flag real problems.\n\n'
            f"=== ANALYSES ===\n{joined}\n=== END ==="
        )
        parsed = _parse_json(llm.invoke(prompt).content)
        issues = parsed.get("issues", []) if parsed.get("verdict") == "revise" else []
        return {
            "issues": issues,
            "revision_count": state.get("revision_count", 0) + (1 if issues else 0),
        }

    return critic


# --------------------------------------------------------------------------
# Reviser node  (acts on Critic feedback)
# --------------------------------------------------------------------------
def make_reviser(retriever, llm):
    """Re-run the readers for each flagged section, with feedback attached."""
    def reviser(state):
        analyses = dict(state["analyses"])
        for issue in state["issues"]:
            section = issue["section"]
            reader = make_reader(section, retriever, llm)
            # reader reads `issues` from state to pick up the feedback
            update = reader({"analyses": analyses, "issues": state["issues"]})
            analyses.update(update["analyses"])
        return {"analyses": analyses}

    return reviser


# --------------------------------------------------------------------------
# Synthesizer node
# --------------------------------------------------------------------------
def make_synthesizer(llm):
    def synthesizer(state):
        analyses = state["analyses"]
        joined = "\n\n".join(f"## {k}\n{v}" for k, v in analyses.items())
        prompt = (
            "You are the Synthesizer. Merge these four validated analyses "
            "into a crisp statement of the paper's core contributions "
            "(3-5 bullet points), then one paragraph on what makes the "
            "approach novel.\n\n"
            f"=== ANALYSES ===\n{joined}\n=== END ==="
        )
        return {"synthesis": llm.invoke(prompt).content}

    return synthesizer


# --------------------------------------------------------------------------
# Architect node
# --------------------------------------------------------------------------
def make_architect(llm):
    def architect(state):
        feedback = state.get("review_feedback", "")
        revise_note = (
            f"\nA reviewer rejected your previous design:\n{feedback}\n"
            f"Produce an improved version addressing this."
            if feedback else ""
        )
        prompt = (
            "You are the Architect. Based on the paper's core contributions "
            "below, propose a concrete system design that APPLIES this "
            "paper's ideas to a real engineering problem. Include: "
            "(1) the problem it solves, (2) component breakdown, "
            "(3) data flow, (4) which paper technique maps to which "
            f"component.{revise_note}\n\n"
            f"=== CONTRIBUTIONS ===\n{state['synthesis']}\n=== END ==="
        )
        return {"design": llm.invoke(prompt).content}

    return architect


# --------------------------------------------------------------------------
# Reviewer node
# --------------------------------------------------------------------------
def make_reviewer(llm):
    def reviewer(state):
        prompt = (
            "You are the Design Reviewer. Judge whether the system design "
            "below is technically sound, faithful to the paper, and "
            "actually implementable. Respond ONLY with JSON:\n"
            '{"verdict": "approve" | "revise", "feedback": "<specific notes>"}\n'
            "Approve unless there is a real flaw.\n\n"
            f"=== DESIGN ===\n{state['design']}\n=== END ==="
        )
        parsed = _parse_json(llm.invoke(prompt).content)
        approved = parsed.get("verdict") == "approve"
        return {
            "review_feedback": "" if approved else parsed.get("feedback", ""),
            "design_revision_count": state.get("design_revision_count", 0)
            + (0 if approved else 1),
        }

    return reviewer

def make_coder(llm):
    def coder(state):
        prompt = (
            "You are the Coder on the team. The system design below has been "
            "approved. Produce the KEY Python components that implement it.\n\n"
            "Requirements:\n"
            "- Write real, idiomatic Python (not pseudocode).\n"
            "- Cover the main classes/functions from the design's component "
            "breakdown — each as a concrete code block.\n"
            "- Add a short docstring/comment to every class and function "
            "explaining its role and how it maps to the paper's technique.\n"
            "- Where a piece needs external logic you won't fully implement, "
            "leave a clearly marked `# TODO:` with a one-line explanation.\n"
            "- Start with a one-line summary comment of the module layout.\n\n"
            "Return ONLY the Python code (no prose around it).\n\n"
            f"=== CORE CONTRIBUTIONS ===\n{state['synthesis']}\n\n"
            f"=== APPROVED DESIGN ===\n{state['design']}\n=== END ==="
        )
        code = llm.invoke(prompt).content
        code = code.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        return {"code": code}
    return coder
