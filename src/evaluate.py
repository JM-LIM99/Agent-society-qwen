"""
Evaluation: blind pairwise LLM-as-judge.

Given the paper context and two final outputs (multi-agent vs baseline),
a *different* model scores both on three axes. To defend against position
bias we randomize which output is labeled "A" vs "B" and tell the judge
nothing about which system produced which.

Returned scores are per-axis (1-10) for each system, plus the judge's
short rationale. main.py turns this into the comparison table.
"""
import json
import random

from langchain_groq import ChatGroq

from . import config


AXES = {
    "completeness": "Covers background, methodology, experiments, AND limitations with no major gaps.",
    "faithfulness": "Claims are grounded in the paper; no invented numbers or mechanisms.",
    "design_depth": "Provides a concrete, implementable system design (components, data flow, technique mapping) — not just a summary.",
}


def _parse_json(text: str) -> dict:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        return json.loads(cleaned[start:end + 1])


def evaluate(multi_output: str, baseline_output: str, paper_context: str = "") -> dict:
    judge = ChatGroq(
        model=config.JUDGE_MODEL,
        temperature=config.JUDGE_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
    )

    # --- blind: randomize position ---
    flip = random.random() < 0.5
    a, b = (baseline_output, multi_output) if flip else (multi_output, baseline_output)
    # mapping back: which label is the multi-agent output?
    multi_label = "B" if flip else "A"

    axes_desc = "\n".join(f"- {k}: {v}" for k, v in AXES.items())
    prompt = (
        "You are an impartial judge comparing two analyses of the same "
        "research paper. Score each on a 1-10 scale for each axis:\n"
        f"{axes_desc}\n\n"
        "Respond ONLY with JSON:\n"
        '{"A": {"completeness": int, "faithfulness": int, "design_depth": int}, '
        '"B": {"completeness": int, "faithfulness": int, "design_depth": int}, '
        '"rationale": "<2-3 sentences>"}\n\n'
        f"=== OUTPUT A ===\n{a}\n\n=== OUTPUT B ===\n{b}\n=== END ==="
    )
    scores = _parse_json(judge.invoke(prompt).content)

    # de-anonymize back to system names
    multi_scores = scores[multi_label]
    baseline_scores = scores["A" if multi_label == "B" else "B"]
    return {
        "multi_agent": multi_scores,
        "baseline": baseline_scores,
        "rationale": scores.get("rationale", ""),
        "_position": f"multi-agent shown as {multi_label}",
    }
