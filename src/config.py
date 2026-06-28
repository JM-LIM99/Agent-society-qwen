"""
Central configuration for the Agent Society pipeline.
Everything tunable lives here so the agents/graph code stays clean.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM (Groq) ---
QWEN_API_KEY = os.getenv("QWEN_API_KEY")  # set this in a .env file
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"
LLM_TEMPERATURE = 0.1  # low temp: we want consistent analysis, not creativity

# Judge model: deliberately DIFFERENT from LLM_MODEL so the evaluator
# isn't grading its own family's output (mitigates self-preference bias).
# Swap to any other model available on your Groq account.
JUDGE_MODEL = "qwen-turbo"
JUDGE_TEMPERATURE = 0.0  # judging should be as deterministic as possible

# --- Embeddings (HuggingFace, runs locally, no API key) ---
# all-MiniLM-L6-v2: English, 384-dim, fast. Good for an English paper.
# If you ever feed a multilingual corpus, swap to
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2".
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64  # ~12% overlap keeps sentences from being cut mid-thought

# --- Vector store ---
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "paper_chunks"

# --- Retrieval ---
RETRIEVAL_K = 5        # chunks returned per query
RETRIEVAL_FETCH_K = 20  # candidates before MMR re-ranking
MMR_LAMBDA = 0.5        # 0 = max diversity, 1 = max relevance
