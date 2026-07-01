"""
FastAPI wrapper around the Agent Society pipeline.

Exposes PDF upload as an HTTP endpoint. Two ways to use it:
  - POST /upload   just stores the PDF and returns its id (fast, no LLM calls).
  - POST /analyze  stores the PDF and runs the full society + baseline + judge
                    pipeline synchronously, returning the report as JSON.
"""
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from .agents import baseline, judge
from .graph import build_graph
from .ingest import load_pdf
from .state import AgentState

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Agent Society API")


def _save_upload(file: UploadFile) -> Path:
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    safe_name = Path(file.filename).name  # strip any path components
    dest = DATA_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            out.write(chunk)
    return dest


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Store a PDF and return its id, without running the pipeline."""
    dest = _save_upload(file)
    return {"file_id": dest.name, "path": str(dest)}


@app.post("/analyze")
def analyze_pdf(file: UploadFile = File(...)):
    """Store a PDF and run the full multi-agent pipeline against it."""
    dest = _save_upload(file)

    try:
        paper = load_pdf(str(dest))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

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

    return {
        "file_id": dest.name,
        "wall_clock_seconds": {"multi_agent": t_multi, "baseline": t_base},
        "judge_verdict": verdict,
        "multi_agent_design": multi_output,
        "baseline": base_output,
    }


@app.get("/reports/{file_id}", response_class=PlainTextResponse)
def get_report(file_id: str):
    """Fetch a previously uploaded PDF's extracted text (debug helper)."""
    path = DATA_DIR / file_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return load_pdf(str(path))
