"""
Single-agent baseline.

One LLM, one prompt, same retrieved context budget as the team gets.
This is the honest comparison point: not a crippled prompt, but a
competent single agent asked to do the whole job at once. The whole
point of the project is to show the agent SOCIETY beats this.
"""
from langchain_groq import ChatGroq

from . import config
from .state import SECTIONS


def run_baseline(retriever):
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.GROQ_API_KEY,
    )

    # Give the baseline the SAME context the four readers would collectively
    # see, so the comparison is about coordination, not information access.
    context_blocks = []
    for section, query in SECTIONS.items():
        docs = retriever.invoke(query)
        context_blocks.append(
            f"[{section}]\n" + "\n".join(d.page_content for d in docs)
        )
    context = "\n\n".join(context_blocks)

    prompt = (
        "You are an expert researcher. Read the paper excerpts below and "
        "produce, in one pass:\n"
        "1. A complete analysis covering background, methodology, "
        "experiments/results, and limitations.\n"
        "2. The paper's core contributions.\n"
        "3. A concrete system design that applies the paper's ideas to a "
        "real engineering problem (components, data flow, technique mapping).\n\n"
        f"=== EXCERPTS ===\n{context}\n=== END ===\n\n"
        "Full analysis and design:"
    )
    print("[baseline] running single-agent pass...")
    return llm.invoke(prompt).content
