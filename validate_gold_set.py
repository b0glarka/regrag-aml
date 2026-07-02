"""Validate the gold set while authoring it.

    python validate_gold_set.py

Reports errors (which must be fixed before the eval) and advisory warnings.
"""

from __future__ import annotations

import sys

from regrag import goldset


def main() -> None:
    errors, warnings = goldset.validate()

    for w in warnings:
        print(f"WARN   {w}")
    for e in errors:
        print(f"ERROR  {e}")
    print()

    if errors:
        print(f"{len(errors)} error(s), {len(warnings)} warning(s). Fix errors before the eval.")
        sys.exit(1)

    items = goldset.load()
    answerable = sum(1 for x in items if x.expects_answer)
    out = sum(1 for x in items if x.is_out_of_scope)
    print(f"OK: {len(items)} items ({answerable} answerable, {out} out-of-scope), no errors.")
    if warnings:
        print(f"{len(warnings)} warning(s) above are advisory, not blocking.")


if __name__ == "__main__":
    main()
