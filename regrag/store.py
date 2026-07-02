"""Chroma vector store (persisted locally).

We compute embeddings ourselves (see regrag.embed) rather than letting Chroma
embed, so the collection is created without an embedding function and we pass
vectors explicitly on add and query. The collection uses cosine space; Chroma
returns cosine *distance*, which we convert to a similarity in [-1, 1].
"""

from __future__ import annotations

import chromadb

from regrag import config


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(reset: bool = False):
    client = get_client()
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass  # collection did not exist yet
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(collection, chunks: list[dict], embeddings: list[list[float]]) -> None:
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"page": c["page"]} for c in chunks],
    )


def search(collection, query_embedding: list[float], k: int) -> list[dict]:
    """Return the top-k hits as {"id", "text", "page", "similarity"}."""
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits: list[dict] = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for id_, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(
            {
                "id": id_,
                "text": doc,
                "page": meta["page"],
                "similarity": 1.0 - dist,  # cosine distance -> similarity
            }
        )
    return hits
