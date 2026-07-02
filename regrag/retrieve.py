"""Retrieval: embed a query and fetch the most similar corpus chunks."""

from __future__ import annotations

from regrag import config, store
from regrag.embed import embed_query


def retrieve(query: str, k: int | None = None) -> list[dict]:
    """Return the top-k chunks for a query as {"id", "text", "page", "similarity"}."""
    k = k or config.TOP_K
    collection = store.get_collection(reset=False)
    query_vector = embed_query(query)
    return store.search(collection, query_vector, k)


def max_similarity(hits: list[dict]) -> float:
    """Top similarity among hits (used for the abstain decision); 0.0 if empty."""
    return max((h["similarity"] for h in hits), default=0.0)


if __name__ == "__main__":
    demo_queries = [
        "What must financial institutions do for customer due diligence?",
        "What is the capital of France?",  # out of scope on purpose
    ]
    for q in demo_queries:
        print(f"\nQuery: {q}")
        hits = retrieve(q)
        for h in hits:
            preview = h["text"][:90].replace("\n", " ").encode("ascii", "replace").decode()
            print(f"  p{h['page']:>3}  sim={h['similarity']:.3f}  {preview}")
        print(f"  -> max similarity: {max_similarity(hits):.3f}")
