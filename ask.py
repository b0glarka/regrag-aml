"""Ask RegRAG a question from the command line.

    python ask.py "What must financial institutions do for customer due diligence?"

With no argument it prompts for a question. Answers cite source pages inline and
abstain when the corpus does not cover the question.
"""

from __future__ import annotations

import sys

from regrag import config
from regrag.generate import answer

# Print UTF-8 so a legacy Windows console cannot crash on regulatory wording
# (curly quotes, section symbols, etc.).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or input("Question: ").strip()
    if not question:
        print("No question given.")
        return

    result = answer(question)

    print()
    if result.abstained:
        print(
            f"[abstained: top similarity {result.max_similarity:.3f} "
            f"< threshold {config.ABSTAIN_SIMILARITY_THRESHOLD:.2f}]"
        )
        print(result.answer)
        return

    print(result.answer)
    pages = ", ".join(f"p. {p}" for p in result.cited_pages)
    print(f"\nRetrieved pages: {pages}  (top similarity {result.max_similarity:.3f})")


if __name__ == "__main__":
    main()
