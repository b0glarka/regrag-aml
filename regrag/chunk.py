"""Chunk the corpus into overlapping, page-scoped passages for retrieval.

Chunks are scoped to a single page so each one cites exactly one source page.
Windows overlap by CHUNK_OVERLAP characters and snap to whitespace so a chunk
never cuts a word in half.
"""

from __future__ import annotations

from regrag import config
from regrag.ingest import load_pages


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, snapped to word breaks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Back off to the last whitespace in the window to avoid cutting a
            # word; if there is none, keep the hard cut.
            ws = text.rfind(" ", start, end)
            if ws > start:
                end = ws
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        # Guarantee forward progress even after a whitespace back-off.
        start = next_start if next_start > start else end
    return chunks


def chunk_pages(
    pages: list[dict] | None = None,
    size: int | None = None,
    overlap: int | None = None,
) -> list[dict]:
    """Return chunk records: {"id", "page", "text"}.

    `id` is stable across runs (e.g. "p015-c00"), so re-indexing the same corpus
    produces the same chunk ids.
    """
    pages = pages if pages is not None else load_pages()
    size = size or config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

    chunks: list[dict] = []
    for rec in pages:
        for i, piece in enumerate(_window(rec["text"], size, overlap)):
            chunks.append(
                {
                    "id": f"p{rec['page']:03d}-c{i:02d}",
                    "page": rec["page"],
                    "text": piece,
                }
            )
    return chunks


if __name__ == "__main__":
    chunks = chunk_pages()
    lengths = [len(c["text"]) for c in chunks]
    avg = sum(lengths) / len(lengths) if lengths else 0
    print(
        f"Built {len(chunks)} chunks "
        f"(avg {avg:.0f} chars, min {min(lengths)}, max {max(lengths)})."
    )
    if chunks:
        sample = next((c for c in chunks if c["page"] == 15), chunks[0])
        safe = sample["text"][:300].replace("\n", " ").encode("ascii", "replace").decode()
        print(f"\nSample chunk {sample['id']} (page {sample['page']}):\n{safe}...")
