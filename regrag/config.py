"""Central configuration for RegRAG-AML.

All tunable knobs live here so the rest of the pipeline reads settings rather
than hardcoding them. Scope discipline: one corpus, one embedding model, one
vector store.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
# repo root = one level up from this file (regrag/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PDF_PATH = RAW_DIR / "fatf_recommendations_2012.pdf"
CHROMA_DIR = REPO_ROOT / ".chroma"
GOLD_SET_PATH = REPO_ROOT / "eval" / "gold_set.yaml"

# --- Corpus ----------------------------------------------------------------
# FATF "The FATF Recommendations" (the 40 Recommendations). The download URLs
# and manual-fallback source page live in download_corpus.py.
CORPUS_NAME = "FATF Recommendations (40 Recommendations)"

# Where to send users for authoritative guidance and for questions this
# assistant cannot or should not answer.
FATF_WEBSITE = "https://www.fatf-gafi.org/"

# SHA-256 of the corpus PDF, for reproducible pinning. The download step
# verifies every run against it; a mismatch means the source bytes changed.
# Pinned from the 2026-06-24 Wayback snapshot of the FATF Recommendations.
CORPUS_SHA256 = "ba862e1f095bceac22ab9a48863e6e8238016a2ebd61c71718febb5c29c285ab"

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
# Chunks retrieved per query. The corpus interleaves terse Recommendation
# statements with long Interpretive Notes; the dense notes crowd the top-5, so
# k=8 gives the retriever room to surface both for a given Recommendation.
TOP_K = 8
# Below this max similarity, the retrieval gate abstains before calling the LLM.
# Tuned on the gold set: answerable questions score >= 0.673, so 0.66 keeps the
# cheap gate from ever refusing a real question. Domain-adjacent out-of-scope
# questions (the gold traps) score 0.67-0.69 and intentionally pass this gate;
# the LLM's scope-refusal prompt (see generate.py) is the semantic backstop for
# those. The gate exists to cheaply catch clearly off-topic queries (< 0.66).
ABSTAIN_SIMILARITY_THRESHOLD = 0.66

# --- Generation ------------------------------------------------------------
# Provider-agnostic: the LLMClient targets any OpenAI-compatible endpoint.
# OpenRouter is the default gateway (one key, many models); Together and any
# other OpenAI-compatible provider are one config change away.
GENERATOR_PROVIDER = os.getenv("REGRAG_GENERATOR_PROVIDER", "openrouter")

# Model IDs are provider-specific (same Llama 3.3 70B, different vendor slug).
DEFAULT_GENERATOR_MODELS = {
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}
GENERATOR_MODEL = os.getenv(
    "REGRAG_GENERATOR_MODEL",
    DEFAULT_GENERATOR_MODELS.get(GENERATOR_PROVIDER, ""),
)

# OpenAI-compatible base URLs and key env vars per provider.
PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
}
PROVIDER_API_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "together": "TOGETHER_API_KEY",
}

# Reproducible eval on OpenRouter: pin the upstream provider and disable
# fallback routing so a run cannot silently switch vendors mid-eval. Leave
# OPENROUTER_PIN_PROVIDER empty for interactive use (let OpenRouter choose).
OPENROUTER_PIN_PROVIDER = os.getenv("REGRAG_OPENROUTER_PROVIDER", "")
OPENROUTER_ALLOW_FALLBACKS = os.getenv("REGRAG_OPENROUTER_ALLOW_FALLBACKS", "0") == "1"

# --- Faithfulness judge ----------------------------------------------------
# Reuses the capstone's production-recommended judge (Haiku 4.5 + versioned
# rubric). The rubric version is recorded alongside every judged result.
JUDGE_MODEL = os.getenv("REGRAG_JUDGE_MODEL", "claude-haiku-4-5")
JUDGE_RUBRIC_VERSION = "v1.0"
