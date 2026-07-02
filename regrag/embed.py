"""Local embeddings via sentence-transformers (no API cost).

bge-style retrieval models expect a short instruction prepended to the query
(not to passages). We handle that asymmetry here so retrieval quality matches
the model's intended use. Embeddings are L2-normalized so cosine similarity
reduces to a dot product.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from regrag import config

# bge-* retrieval instruction, prepended to QUERIES only.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load (and cache) the embedding model. First call downloads the weights."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


def _uses_bge_instruction() -> bool:
    return config.EMBEDDING_MODEL.lower().startswith("baai/bge")


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed corpus passages (no query instruction)."""
    model = get_model()
    vectors = model.encode(
        list(texts), normalize_embeddings=True, convert_to_numpy=True
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a search query (with the bge instruction when applicable)."""
    model = get_model()
    prepared = _QUERY_INSTRUCTION + text if _uses_bge_instruction() else text
    vector = model.encode(
        [prepared], normalize_embeddings=True, convert_to_numpy=True
    )[0]
    return vector.tolist()
