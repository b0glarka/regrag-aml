"""Build the vector store: chunk the corpus, embed locally, load into Chroma.

Run from the repo root:
    python build_index.py

The first run downloads the embedding model weights (a few hundred MB). The
Chroma store is written to .chroma/ (gitignored) and is rebuilt from scratch
each run so the index always matches the current corpus and chunking.
"""

from __future__ import annotations

from regrag import config, store
from regrag.chunk import chunk_pages
from regrag.embed import embed_passages

# Chroma add batch size (keeps memory flat on large corpora).
_BATCH = 128


def main() -> None:
    chunks = chunk_pages()
    print(f"Chunks to index: {len(chunks)}")

    print(f"Embedding with {config.EMBEDDING_MODEL} (first run downloads weights)...")
    embeddings = embed_passages([c["text"] for c in chunks])

    collection = store.get_collection(reset=True)
    for i in range(0, len(chunks), _BATCH):
        store.add_chunks(collection, chunks[i : i + _BATCH], embeddings[i : i + _BATCH])

    print(f"Indexed {collection.count()} chunks into '{config.COLLECTION_NAME}'")
    print(f"Vector store: {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
