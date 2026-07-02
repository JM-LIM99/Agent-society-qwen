"""
Builds the markdown report shared by the CLI (main.py) and the API (api.py).
"""
import json
import time

from .agents import baseline, judge
from .graph import build_graph
from .ingest import load_pdf
from .state import AgentState


def build_report(pdf_path: str, source_name: str) -> str:
    """Run the full pipeline against a PDF and return the report as markdown."""
    paper = load_pdf(pdf_path)
    init: AgentState = {
        "paper_text": paper, "analyses": {}, "critique": "",
        "revisions": 0, "design": "", "review": "",
        "design_revisions": 0, "code": "",
    }

    t0 = time.time()
    final = build_graph().invoke(init)
    multi_output = final["design"] + "\n\n# CODE\n" + final["code"]
    t_multi = time.time() - t0

    t0 = time.time()
    base_output = baseline(paper)
    t_base = time.time() - t0

    verdict = judge(multi_output, base_output, paper)

    lines = [
        "# Agent Society — Report\n",
        f"Paper: `{source_name}`\n",
        f"Wall-clock: multi {t_multi:.1f}s, baseline {t_base:.1f}s\n",
        "## Judge verdict\n```json",
        json.dumps(verdict, indent=2),
        "```\n\n## Multi-agent design",
        multi_output,
        "\n\n## Baseline",
        base_output,
    ]
    return "\n".join(lines)
