"""
End-to-end runner.

    python -m src.main data/attention.pdf

Steps:
  1. Ensure the paper is ingested (embeds once if not).
  2. Run the multi-agent graph  -> design report.
  3. Run the single-agent baseline.
  4. Blind pairwise judge -> comparison table.
  5. Write everything to report.md.
"""
import os
import sys
import time

from . import config
from .ingest import ingest, get_retriever
from .agents import make_llm
from .graph import build_graph
from .baseline import run_baseline
from .evaluate import evaluate, AXES


def _ensure_ingested(pdf_path):
    if not os.path.isdir(config.CHROMA_DIR) or not os.listdir(config.CHROMA_DIR):
        print("[main] no vector store found, ingesting...")
        ingest(pdf_path)
    else:
        print("[main] vector store exists, skipping ingest.")


def _run_multi_agent(retriever, llm):
    app = build_graph(retriever, llm)
    initial = {
        "analyses": {}, "issues": [], "revision_count": 0,
        "synthesis": "", "design": "", "review_feedback": "",
        "design_revision_count": 0,
    }
    print("[main] running multi-agent society...")
    final = app.invoke(initial)
    # final report = synthesis + design
    return (
        "## Core Contributions\n" + final["synthesis"]
        + "\n\n## Proposed System Design\n" + final["design"]
    ), final


def _table(result):
    rows = ["| Axis | Multi-Agent | Baseline | Δ |",
            "|---|---|---|---|"]
    totals = {"multi": 0, "base": 0}
    for axis in AXES:
        m = result["multi_agent"][axis]
        b = result["baseline"][axis]
        totals["multi"] += m
        totals["base"] += b
        rows.append(f"| {axis} | {m} | {b} | {m - b:+d} |")
    rows.append(f"| **total** | **{totals['multi']}** | **{totals['base']}** "
                f"| **{totals['multi'] - totals['base']:+d}** |")
    gain = (totals["multi"] - totals["base"]) / max(totals["base"], 1) * 100
    rows.append(f"\n**Efficiency gain: {gain:+.1f}%** over single-agent baseline.")
    return "\n".join(rows)


def main(pdf_path):
    _ensure_ingested(pdf_path)
    retriever = get_retriever()
    llm = make_llm()

    t0 = time.time()
    multi_report, _ = _run_multi_agent(retriever, llm)
    t_multi = time.time() - t0

    t0 = time.time()
    baseline_report = run_baseline(retriever)
    t_baseline = time.time() - t0

    print("[main] judging...")
    result = evaluate(multi_report, baseline_report)

    report = (
        f"# Agent Society — Analysis Report\n\n"
        f"Paper: `{os.path.basename(pdf_path)}`\n\n"
        f"## Efficiency Comparison\n{_table(result)}\n\n"
        f"_Judge: {config.JUDGE_MODEL} · {result['_position']}_\n"
        f"_Wall-clock: multi-agent {t_multi:.1f}s, baseline {t_baseline:.1f}s_\n\n"
        f"Judge rationale: {result['rationale']}\n\n"
        f"---\n\n# Multi-Agent Output\n{multi_report}\n\n"
        f"---\n\n# Single-Agent Baseline Output\n{baseline_report}\n"
    )

    out_path = os.path.join(os.path.dirname(__file__), "..", "report.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\n[main] done -> {out_path}")
    print("\n" + _table(result))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.main <path_to_pdf>")
        sys.exit(1)
    main(sys.argv[1])
