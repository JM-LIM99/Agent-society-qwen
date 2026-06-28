"""
Ingestion pipeline.

    PDF  ->  text  ->  512-token chunks  ->  HF embeddings  ->  ChromaDB

Run this once per paper. It persists the vector store to disk so the
agent graph can attach to it later without re-embedding.

Usage:
    python -m src.ingest data/attention.pdf
"""
import sys
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from . import config


def load_pdf(pdf_path: str):
    """Load a PDF into a list of page-level Documents (one per page)."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[ingest] loaded {len(pages)} pages from {os.path.basename(pdf_path)}")
    return pages

def chunk_documents(pages):
    """
    Split pages into ~512-char chunks.

    Note: RecursiveCharacterTextSplitter counts *characters*, not model
    tokens. For a quick hackathon baseline that's fine. If you want true
    token-based 512 chunks, swap to
    RecursiveCharacterTextSplitter.from_huggingface_tokenizer(...).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        # try to split on natural boundaries first, fall back to chars
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    print(f"[ingest] produced {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks with a local HF model and persist to ChromaDB."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.CHROMA_DIR,
    )
    print(f"[ingest] embedded + stored {len(chunks)} chunks "
          f"-> {config.CHROMA_DIR}")
    return vectorstore


def get_retriever():
    """
    Re-attach to an already-built ChromaDB collection and return an
    MMR retriever. The agent graph calls this; it does NOT re-embed.
    """
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.RETRIEVAL_K,
            "fetch_k": config.RETRIEVAL_FETCH_K,
            "lambda_mult": config.MMR_LAMBDA,
        },
    )


def ingest(pdf_path: str):
    """Full pipeline: load -> chunk -> embed -> store."""
    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages)
    build_vectorstore(chunks)
    print("[ingest] done. Vector store ready for the agent graph.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.ingest <path_to_pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
