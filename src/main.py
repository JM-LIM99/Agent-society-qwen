
import sys
import os
import glob
from dotenv import load_dotenv
from . import config
from .report import build_report

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def find_default_pdf() -> str:
    """Pick the first PDF in data/ when no path is given on the command line."""
    pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {DATA_DIR}")
    return pdfs[0]

def main(pdf_path: str):
    report = build_report(pdf_path, os.path.basename(pdf_path))
    with open("report.md", "w") as f:
        f.write(report)
    print("\n[done] wrote report.md")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        pdf_path = sys.argv[1]
    elif len(sys.argv) == 1:
        pdf_path = find_default_pdf()
        print(f"[main] no path given, using {pdf_path}")
    else:
        print("Usage: python -m src.main [path_to_pdf]")
        sys.exit(1)
    main(pdf_path)
