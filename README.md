# Agent Society — Paper → System Design

Multi-agent collaboration system (Track 3). A society of specialized agents
reads a research paper, resolves disagreements through a critique loop, and
produces a concrete system design — measurably better than a single agent
doing the same job alone.

## What it shows (Track 3 criteria)

1. **Task decomposition & roles** — a paper is split across four fixed Reader
   agents (background / methodology / experiments / limitations), each pulling
   its own context from the vector store.
2. **Disagreement & conflict resolution** — a **Critic** cross-checks the four
   analyses for contradictions and unsupported claims, and routes flagged
   sections back to a **Reviser** (bounded loop). A second loop lets the
   **Reviewer** bounce a weak design back to the **Architect**.
3. **Measurable efficiency gain** — a blind pairwise LLM-as-judge (a *different*
   model) scores the multi-agent output vs a single-agent baseline on
   completeness, faithfulness, and design depth.

## Architecture

```
PDF ─► ingest (512-char chunks ─► HF embeddings ─► ChromaDB)
                                      │
                                      ▼
  read_background ─► read_methodology ─► read_experiments ─► read_limitations
                                      │
                                      ▼
                  ┌──────────────► critic ──(issues)──► reviser ─┐
                  └──(approved)◄────────────────────────────────┘
                                      │
                                      ▼
                  synthesizer ─► architect ─► reviewer ──(approve)──► coder ──► END
                                                            │
                                                (revise)──► architect
                                                
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
```

## Run

```bash
# put your paper here, e.g. Attention Is All You Need
#   data/attention.pdf

python -m src.ingest data/attention.pdf   # one-time: embed the paper
python -m src.main data/attention.pdf     # run society + baseline + judge
```

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
| `config.py` | all tunable params (models, chunk size, retrieval) |
| `ingest.py` | PDF → chunks → embeddings → ChromaDB |
| `state.py` | shared blackboard + the four reader roles |
| `agents.py` | reader / critic / reviser / synthesizer / architect / reviewer |
| `graph.py` | LangGraph wiring + the two rework loops |
| `baseline.py` | single-agent comparison point |
| `evaluate.py` | blind pairwise LLM-as-judge |
| `main.py` | end-to-end runner |

## Tuning knobs to mention in the demo

- Swap `EMBEDDING_MODEL` to the multilingual MiniLM for non-English papers.
- `MAX_REVISIONS` / `MAX_DESIGN_REVISIONS` trade quality vs cost.
- `JUDGE_MODEL` must differ from `LLM_MODEL` to keep the eval honest.
# Agent-society-qwen
