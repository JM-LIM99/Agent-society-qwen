import sys
import os
from langchain_community.document_loaders import PyPDFLoader 

def load_pdf(pdf_path: str) -> str:
    """Load a PDF file into text."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found : {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n\n".join(page.page_content for page in pages)
    print(f"[ingest] loaded {len(pages)} pages, {len(full_text)} chars"
          f"from {os.path.basename(pdf_path)}")
    return full_text

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m scr.ingest <path_to_pdf>")
        sys.exit(1)
    text = load_pdf(sys.argv[1])
    print(text[:500])