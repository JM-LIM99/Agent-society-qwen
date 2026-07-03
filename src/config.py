"""
Central configuration for the Agent Society pipeline.
Everything tunable lives here so the agents/graph code stays clean.
"""
import os
from dotenv import load_dotenv

load_dotenv()


QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

LLM_MODEL = "qwen-plus-latest"
LLM_CODER = "qwen3-coder-plus"
JUDGE_MODEL = "qwen3-30b-a3b-instruct-2507"
MAX_REVISIONS = 1
MAX_DESIGN_REVISIONS = 1

READER_ROLES = ["background", "methodology", "experiments", "limitations"]
