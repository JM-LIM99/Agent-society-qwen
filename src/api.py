"""
FastAPI wrapper around the Agent Society pipeline.

One endpoint: POST /analyze with a PDF, get a .md report back.
"""
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from .report import build_report

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Agent Society API")


@app.post("/analyze")
def analyze_pdf(file: UploadFile = File(...)):
    """Take a PDF, run the full multi-agent pipeline, and return the report as markdown."""
    if file.content_type != "application/pdf" and not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    safe_name = Path(file.filename).name  # strip any path components
    dest = DATA_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            out.write(chunk)

    try:
        report = build_report(str(dest), safe_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    md_name = f"{Path(safe_name).stem}_report.md"
    return Response(
        content=report,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{md_name}"'},
    )
