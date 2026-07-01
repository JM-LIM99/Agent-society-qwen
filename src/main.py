
import sys
import os
import json
import time
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from . import config
from .ingest import load_pdf
from .agents import baseline, judge
from .graph import build_graph
from .state import AgentState
from dotenv import load_dotenv

load_dotenv()

def main(pdf_path: str):
    paper = load_pdf(pdf_path)
    init: AgentState = {
        "paper_text": paper, "analyses": {}, "critique": "",
        "revisions": 0, "design": "", "review": "",
        "design_revisions": 0, "code": "",
    }

    print("\n=== MULTI-AGENT ===")
    t0 = time.time()
    final = build_graph().invoke(init)
    multi_output = final["design"] + "\n\n# CODE\n" + final["code"]
    t_multi = time.time() - t0

    print("\n=== BASELINE ===")
    t0 = time.time()
    base_output = baseline(paper)
    t_base = time.time() - t0

    print("\n=== JUDGE ===")
    verdict = judge(multi_output, base_output, paper)

    with open("report.md", "w") as f:
        f.write("# Agent Society — Report\n\n")
        f.write(f"Paper: `{os.path.basename(pdf_path)}`\n\n")
        f.write(f"Wall-clock: multi {t_multi:.1f}s, baseline {t_base:.1f}s\n\n")
        f.write("## Judge verdict\n```json\n")
        f.write(json.dumps(verdict, indent=2))
        f.write("\n```\n\n## Multi-agent design\n")
        f.write(multi_output)
        f.write("\n\n## Baseline\n")
        f.write(base_output)
    print("\n[done] wrote report.md")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_pdf>")
        sys.exit(1)
    main(sys.argv[1])
