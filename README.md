# Qwen-Powered Agent Society for Document Analysis

Multi-agent collaboration system (Track 3). A society of specialized agents
reads a research paper (full text, no retrieval), resolves disagreements
through a critique loop, and produces a concrete system design — measurably
better than a single agent doing the same job alone.

## What it shows (Track 3 criteria)

1. **Task decomposition & roles** — a paper is split across four fixed Reader
   agents (background / methodology / experiments / limitations), each given
   the full paper text but instructed to focus only on its own section.
2. **Disagreement & conflict resolution** — a **Critic** cross-checks the four
   analyses for contradictions and unsupported claims, and routes flagged
   sections back to a **Reviser** (bounded loop). A second loop lets the
   **Reviewer** bounce a weak design back to the **Architect**.
3. **Measurable efficiency gain** — a blind pairwise LLM-as-judge (a *different*
   model) scores the multi-agent output vs a single-agent baseline on
   completeness, faithfulness, and design depth.
4. **Design-to-code** — once the design is approved, a **Coder** agent turns it
   into runnable Python with docstrings mapping each part back to the paper's
   technique.

## Architecture

```
PDF ─► load_pdf (main.py, plain text — no chunking/embedding)
         │
         ▼
  read_background ─► read_methodology ─► read_experiments ─► read_limitations
                                      │
                                      ▼
                  ┌──────────────► critic ──(issues)──► reviser ─┐
                  └──(approved)◄────────────────────────────────┘
                                      │
                                      ▼
                                synthesizer ─► explainer
                                                  │
                                                  ▼
                  architect ──► reviewer ──(approve)──► coder ──► END
                     ▲              │
                     └──(revise)────┘
```

`explainer` produces a plain-language summary of the paper alongside the
technical synthesis; `synthesizer` and `explainer` both feed the `architect`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add QWEN_API_KEY=<your dashscope key>
```

Models are served through Qwen's DashScope OpenAI-compatible endpoint
(`QWEN_BASE_URL` in `src/config.py`), so `langchain-openai`'s `ChatOpenAI`
client is reused with a custom `base_url`/`api_key` rather than calling
OpenAI directly.

## Run

```bash
# put your paper here, e.g. Attention Is All You Need
#   data/attention.pdf

python -m src.main               # auto-picks the first PDF found in data/
python -m src.main data/foo.pdf  # or run against a specific PDF
```

There's no separate embedding/ingest step — the whole paper is passed as
plain text to every reader. `src/ingest.py` only exposes a standalone
`load_pdf` helper you can run directly (`python -m src.ingest data/attention.pdf`)
to preview extracted text; it isn't part of the pipeline.

Output is written to `report.md`: the comparison table, the multi-agent
design, and the baseline — side by side.

## Why multi-agent wins (the honest framing)

The society is *slower* in wall-clock time (agents run sequentially). The win
is **quality per task**, not speed:
- Dedicated readers → nothing gets skipped (completeness).
- The Critic filters hallucinated claims (faithfulness).
- The Architect/Reviewer loop reaches a real design; a single pass usually
  stops at a summary (design depth).

The judge is a *different* model, and outputs are shown to it in **random A/B
order**, so neither self-preference nor position bias inflates the result.

## Files

| File | Role |
|---|---|
| `config.py` | LLM models (`LLM_MODEL`, `LLM_CODER`, `JUDGE_MODEL`), revision limits, reader roles |
| `ingest.py` | standalone PDF → text helper (debug/preview only, not wired into the graph) |
| `state.py` | shared blackboard (`AgentState`), the four reader roles (`SECTIONS`), loop guard constants |
| `agents.py` | all agent nodes — readers, critic, reviser, synthesizer, explainer, architect, reviewer, coder — plus the `baseline` agent and the `judge` evaluator |
| `graph.py` | LangGraph wiring + the two rework loops |
| `main.py` | end-to-end runner: loads the PDF, runs the graph, runs the baseline, runs the judge, writes `report.md` |

## Tuning knobs to mention in the demo

- `LLM_MODEL` / `LLM_CODER` pick the reasoning vs. code-generation models.
- `MAX_REVISIONS` / `MAX_DESIGN_REVISIONS` trade quality vs cost.
- `JUDGE_MODEL` must differ from `LLM_MODEL` to keep the eval honest.
