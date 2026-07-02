"""Central configuration for RegRAG-AML.

All tunable knobs live here so the rest of the pipeline reads settings rather
than hardcoding them. Scope discipline: one corpus, one embedding model, one
vector store.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# repo root = two levels up from this file (src/regrag/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CHROMA_DIR = REPO_ROOT / ".chroma"
GOLD_SET_PATH = REPO_ROOT / "eval" / "gold_set.jsonl"

# --- Corpus ----------------------------------------------------------------
# FATF "The FATF Recommendations" (the 40 Recommendations). The exact download
# URL and file name are documented in the corpus download script.
CORPUS_NAME = "FATF Recommendations (40 Recommendations)"

# --- Chunking --------------------------------------------------------------
CHUNK_SIZE = 900          # characters per chunk (approx.)
CHUNK_OVERLAP = 150       # character overlap between adjacent chunks

# --- Embeddings (local, no API key) ---------------------------------------
# bge-small is a strong small retriever; swap to all-MiniLM-L6-v2 for a lighter
# footprint on memory-constrained deploys.
EMBEDDING_MODEL = os.getenv("REGRAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME = "fatf_recommendations"

# --- Retrieval -------------------------------------------------------------
TOP_K = 5                 # chunks retrieved per query
# Below this max similarity, the assistant abstains ("not covered in these
# documents") instead of answering. Tuned on the gold set on Day 2.
ABSTAIN_SIMILARITY_THRESHOLD = 0.30

# --- Generation ------------------------------------------------------------
# Default provider reuses the capstone's production generator. The LLMClient
# also supports "openrouter" and any OpenAI-compatible endpoint.
GENERATOR_PROVIDER = os.getenv("REGRAG_GENERATOR_PROVIDER", "together")
GENERATOR_MODEL = os.getenv(
    "REGRAG_GENERATOR_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

# OpenAI-compatible base URLs per provider.
PROVIDER_BASE_URLS = {
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
PROVIDER_API_KEY_ENV = {
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# --- Faithfulness judge ----------------------------------------------------
# Reuses the capstone's production-recommended judge (Haiku 4.5 + versioned
# rubric). The rubric version is recorded alongside every judged result.
JUDGE_MODEL = os.getenv("REGRAG_JUDGE_MODEL", "claude-haiku-4-5")
JUDGE_RUBRIC_VERSION = "v1.0"
