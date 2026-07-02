"""Find which corpus pages contain a phrase, to fill in gold-set expected_pages.

    python find_page.py "customer due diligence"

Prints each matching page with a short snippet of context. Use it to confirm
the exact page(s) that answer a question before recording them in the gold set.
"""

from __future__ import annotations

import sys

from regrag.ingest import load_pages

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> None:
    term = " ".join(sys.argv[1:]).strip()
    if not term:
        print('Usage: python find_page.py "search phrase"')
        return

    needle = term.lower()
    matches = 0
    for page in load_pages():
        text = page["text"]
        idx = text.lower().find(needle)
        if idx == -1:
            continue
        matches += 1
        start = max(0, idx - 45)
        end = min(len(text), idx + len(term) + 45)
        snippet = text[start:end].replace("\n", " ")
        print(f"page {page['page']:>3}: ...{snippet}...")

    print()
    if matches:
        print(f'{matches} page(s) contain "{term}".')
    else:
        print(f'No page contains "{term}". Try a shorter or different phrase.')


if __name__ == "__main__":
    main()
