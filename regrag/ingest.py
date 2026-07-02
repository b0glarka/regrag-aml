"""PDF ingestion: extract clean per-page text from the FATF corpus.

Extraction is page-based so that retrieved chunks can cite a source page. Two
cleaning steps run before anything downstream:

1. Private-use-area glyphs (custom PDF symbols like bullets, e.g. U+F8E9) are
   dropped; they carry no textual meaning.
2. Running headers and footers are removed. FATF repeats a header and a
   "<page> (c) 2012-2025" footer on nearly every page; left in, that boilerplate
   would pollute every chunk's embedding and hurt retrieval. We detect it by how
   often a line recurs across pages rather than hardcoding the exact strings, so
   it still works if FATF tweaks the wording.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from regrag import config

# Unicode Private Use Area (U+E000..U+F8FF): custom glyphs with no text meaning.
# Built with chr() so the source stays plain ASCII (no \u escapes to corrupt).
_PUA_RE = re.compile("[" + chr(0xE000) + "-" + chr(0xF8FF) + "]")

# Control characters except tab and newline.
_CTRL_CHARS = (
    "".join(chr(c) for c in range(0x00, 0x09))
    + chr(0x0B)
    + chr(0x0C)
    + "".join(chr(c) for c in range(0x0E, 0x20))
)
_CTRL_RE = re.compile("[" + re.escape(_CTRL_CHARS) + "]")

_DIGITS_RE = re.compile(r"\d+")

# A line is treated as a running header/footer if it repeats on at least this
# fraction of pages.
_BOILERPLATE_PAGE_FRACTION = 0.4


def _normalize_line(line: str) -> str:
    line = _PUA_RE.sub("", line)
    line = _CTRL_RE.sub("", line)
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def _line_key(line: str) -> str:
    """Digit-collapsed form so page-numbered footers all map to one key."""
    return _DIGITS_RE.sub("#", line)


def _find_boilerplate_keys(page_lines: list[list[str]]) -> set[str]:
    """Return the digit-collapsed keys of lines that recur across many pages."""
    counts: Counter[str] = Counter()
    for lines in page_lines:
        # Count each distinct line once per page.
        for key in {_line_key(line) for line in lines}:
            counts[key] += 1
    threshold = max(5, int(_BOILERPLATE_PAGE_FRACTION * len(page_lines)))
    return {key for key, count in counts.items() if count >= threshold}


# Footer years used to locate the printed page number, e.g. "14 (c) 2012-2025".
_FOOTER_YEARS = ("2012", "2025")


def _printed_page_number(raw: str) -> int | None:
    """Extract the document's own printed page number from a page footer.

    The footer reads like "14 (c) 2012-2025" (sometimes right-aligned). We take
    the 1-3 digit number on that line that is not one of the copyright years.
    """
    for line in raw.replace("\r", "\n").split("\n"):
        if "2012" in line and "2025" in line:
            candidates = [
                int(n) for n in re.findall(r"\d{1,3}", line) if n not in _FOOTER_YEARS
            ]
            if candidates:
                return candidates[0]
    return None


def _physical_to_printed_offset(raws: list[str]) -> int:
    """Derive the constant offset between physical PDF index and printed page.

    The FATF PDF prints page N on physical page N+1 (a cover shifts it). We read
    the offset from the footers rather than hardcode it, so it self-corrects if
    the front matter changes. Returns 0 if no printed numbers are found.
    """
    counts: Counter[int] = Counter()
    for i, raw in enumerate(raws, start=1):
        printed = _printed_page_number(raw)
        if printed is not None:
            counts[i - printed] += 1
    return counts.most_common(1)[0][0] if counts else 0


def load_pages(pdf_path: Path | None = None) -> list[dict]:
    """Return the corpus as a list of {"page": int, "text": str} records.

    `page` is the document's own PRINTED page number (from the footer), not the
    physical PDF index, so citations and gold-set page references match what a
    human sees in the FATF PDF. Running headers/footers and empty pages are
    dropped.
    """
    pdf_path = pdf_path or config.RAW_PDF_PATH
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Corpus PDF not found at {pdf_path}. "
            "Run `python download_corpus.py` first."
        )

    reader = PdfReader(str(pdf_path))
    raws = [(page.extract_text() or "").replace("\r", "\n") for page in reader.pages]
    offset = _physical_to_printed_offset(raws)

    # Pass 1: normalize each page into non-empty cleaned lines; label each with
    # the printed page number (physical index minus the derived offset).
    page_labels: list[int] = []
    page_lines: list[list[str]] = []
    for i, raw in enumerate(raws, start=1):
        lines = [_normalize_line(line) for line in raw.split("\n")]
        lines = [line for line in lines if line]
        page_labels.append(i - offset)
        page_lines.append(lines)

    # Pass 2: drop running headers/footers, then keep pages that still have text.
    boilerplate = _find_boilerplate_keys(page_lines)
    records: list[dict] = []
    for label, lines in zip(page_labels, page_lines):
        kept = [line for line in lines if _line_key(line) not in boilerplate]
        text = "\n".join(kept).strip()
        if text:
            records.append({"page": label, "text": text})
    return records


if __name__ == "__main__":
    pages = load_pages()
    total_chars = sum(len(p["text"]) for p in pages)
    print(f"Extracted {len(pages)} non-empty pages, {total_chars:,} characters.")
    if pages:
        first = pages[0]
        # Encode defensively so a legacy (cp1252) console cannot crash on a
        # stray non-ASCII character in the preview.
        preview = first["text"][:300].replace("\n", " ")
        safe = preview.encode("ascii", "replace").decode()
        print(f"\nPage {first['page']} preview:\n{safe}...")
