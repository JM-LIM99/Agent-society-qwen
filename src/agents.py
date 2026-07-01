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
from langchain_openai import OpenAIEmbeddings
import sys
import os
import json
import time
import re, json, random
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from .state import AgentState
from .config import QWEN_API_KEY, QWEN_BASE_URL, LLM_MODEL, JUDGE_MODEL, LLM_CODER

from . import config
from .state import SECTIONS


load_dotenv()


def make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model= model,
        api_key = QWEN_API_KEY,
        base_url = QWEN_BASE_URL,
        temperature = 0, 
        extra_body = {"enable_thinking": False},
    )
# Load paper - no chunk, no embed
def load_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    pages = PyPDFLoader(pdf_path).load()
    text = "\n\n".join(p.page_content for p in pages)
    print(f"[load] {len(pages)} pages, {len(text)} chars")
    return text

llm = make_llm(LLM_MODEL)
coding = make_llm(LLM_CODER)

def read_section(state: AgentState, role: str) -> dict:
    """One reader per section. Gets the WHOLE paper, focuses on its role"""
    paper = state["paper_text"]
    prompt = (
        f"You are a research analyst. Read the paper below and analyze ONLY"
        f"its **{role}**. Be specific and cite concrete details from the text.\n\n"
        f"=== PAPER ===\n{paper}\n=== END ==="
    )
    resp = llm.invoke(prompt).content
    analyses = dict(state.get("analyses", {}))
    analyses[role] = resp
    print(f"[reader:{role}] done ({len(resp)} chars)")
    return {"analyses": analyses}

def critic(state: AgentState) -> dict:
    joined = "\n\n".join(f"## {r}\n{t}" for r, t in state["analyses"].items())
    prompt = (
        "You are a critic. Check these analyses against the paper for accuracy, "
        "missing points, and unsupported claims. If they are solid, reply exactly "
        "'APPROVED'. Otherwise list specific issues to fix.\n\n"
        f"=== ANALYSES ===\n{joined}"
    )
    critique = llm.invoke(prompt).content
    print(f"[critic] {'APPROVED' if 'APPROVED' in critique else 'needs work'}")
    return {"critique": critique}

def reviser(state: AgentState) -> dict:
    joined = "\n\n".join(f"## {r}\n{t}" for r,t in state["analyses"].items()) 
    prompt = (
        f"Revise the analyses to fix these issues:\n{state['critique']}\n\n"
        f"=== CURRENT ANALYSES===\n{joined}\n\n"
        "Return the corrected analyses, same four sections"
    )

    revised = llm.invoke(prompt).content
    analyses = dict(state["analyses"])
    analyses["_reviser"] = revised
    print(f"[reviser] revision {state.get('revisions', 0) + 1}")
    return {"analyses": analyses, "revisions": state.get("revisions", 0) + 1}

def synthesizer(state: AgentState) -> dict:
    joined = "\n\n".join(f"## {r}\n{t}" for r, t in state["analyses"].items())
    prompt = (
        "Synthesize these analyses into the paper's core contributions. "
        "You MUST preserve every specific number, hyperparameter, model name, "
        "dataset, equation, and quantitative result mentioned in the analyses. "
        "Do not drop them for brevity. Bullet count is not limited if numbers "
        "would otherwise be lost.\n\n" + joined
)
    summary = llm.invoke(prompt).content
    print("[synthesizer] done")
    return {"analyses": {**state["analyses"], "_summary": summary}}

EXPLAIN_SYSTEM = (
    "You explain a research paper in plain, simple language.\n"
    "Imagine explaining it to a smart friend who is NOT an expert.\n"
    "Rules:\n"
    "1. Use only what's in the analyses below. Don't invent numbers or outside facts.\n"
    "2. Avoid jargon. If you must use a technical term, explain it in one short phrase.\n"
    "3. Use everyday analogies where they help.\n"
    "4. Short sentences. No marketing words."
)
def explainer(state: AgentState) -> dict:
    joined = "\n\n".join(f"## {r}\n{t}" for r, t in state["analyses"].items())
    user = (
        f"<analyses>\n{joined}\n</analyses>\n\n"
        "Explain this paper simply:\n"
        "1. What problem does it solve? (one plain sentence)\n"
        "2. What's the key idea? (use an analogy if it helps)\n"
        "3. Why does it matter?\n"
        "Use ONLY the text above."
    )
    explanation = llm.invoke([
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": user},
    ]).content
    print("[explainer] done")
    return {"analyses": {**state["analyses"], "_explanation": explanation}}

def architect(state: AgentState) -> dict:
    summary = state["analyses"].get("_summary", "")
    feedback = state.get("review", "")
    extra = f"\nAddress this prior review feedback:\n{feedback}" if feedback else""
    prompt = (
        "Based on this paper summary, design a concrete system that implements"
        "its core method. Give: (1) problem it solves, (2) components"
        f"(3) data flow.{extra}\n\n=== SUMMARY ===\n{summary}"
        "use ONLY the summary above. For any detail not in the summary,"
        f"write '[not specified in summary]'"
    )
    design = llm.invoke(prompt).content

    print("[architect] done")

    return {"design": design}

def reviewer(state: AgentState) -> dict:
    prompt = (
        "Review this system design for soundness. If solid, reply exactly"
        "'APPROVED'. Otherwise give specific fixes.\n\n" 
        + state["design"]
    )
    review = llm.invoke(prompt).content
    print(f"[reviwer] {'APPROVED' if 'APPROVED' in review else 'needs work'}")

    return {"review" : review,
            "design_revisions": state.get("design_revisions", 0) + 1 }

def coder(state: AgentState) -> dict:
    prompt = (
        "Turn this approved design into minimal runnable Python with docstrings "
        "mapping each part back to the paper's technique.\n\n"
        "GROUNDING RULES:\n"
        "- Implement ONLY what the design specifies.\n"
        "- For any value marked '[not specified in summary]', do NOT invent a number. "
        "Make it a constructor argument with NO default, or set it to None with a "
        "comment '# not specified in source'.\n"
        "- Never write a plausible-looking number and claim it matches the paper.\n\n"
        f"=== DESIGN ===\n{state['design']}"
    )
    code = coding.invoke(prompt).content
    print("[coder] done")
    return {"code": code}

def baseline(paper_text: str) -> str:
    prompt = (
        "You are an expert researcher. Read this paper, summarize its core"
        "contributions, and design a system implementing its method.\n\n"
        f"=== PAPER ===\n{paper_text}"
    )
    return make_llm(LLM_MODEL).invoke(prompt).content

def extract_json(raw: str) -> dict:
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    if start == -1:
        raise json.JSONDecodeError("no object", raw, 0)
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{": depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i+1])
    raise json.JSONDecodeError("unbalanced", raw, 0)


def judge(multi: str, base: str, source: str) -> dict:
    judge_llm = make_llm(JUDGE_MODEL)

    multi_is_a = random.random() < 0.5
    a, b = (multi, base) if multi_is_a else (base, multi)
    prompt = (
        "You are comparing two analyses (A and B) of the SAME source paper.\n"
        "Judge ONLY against the SOURCE below. Do not reward fluent writing.\n\n"
        "For EACH analysis, score three axes 1-5:\n"
        "- completeness: covers key points actually in the source.\n"
        "- faithfulness: 5 = every specific claim (numbers, hyperparameters, "
        "architecture details) is supported by the source. Each claim NOT in "
        "the source subtracts 1. Inventing specs = severe penalty. Correctly "
        "writing 'not specified' is NOT penalized.\n"
        "- design_depth: quality of reasoning given only what the source supports.\n\n"
        f"=== SOURCE ===\n{source}\n\n"
        f"=== A ===\n{a}\n\n=== B ===\n{b}\n\n"
        "Reply ONLY JSON:\n"
        '{"A":{"completeness":x,"faithfulness":x,"design_depth":x},'
        '"B":{"completeness":x,"faithfulness":x,"design_depth":x}}'
        )
    raw = judge_llm.invoke(prompt).content
    try:
        result = extract_json(raw)
        a_total = sum(result["A"].values())
        b_total = sum(result["B"].values())
        result["winner"] = "A" if a_total >= b_total else "B"
        result["_multi_is"] = "A" if multi_is_a else "B"
    except (json.JSONDecodeError, KeyError) as e:
        result = {"error": f"judge parse failed: {e}", "raw": raw}
    return result
