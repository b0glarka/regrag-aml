"""Tune the abstain similarity threshold on the gold set.

Retrieval only, no generation or judging, so it costs nothing to run. We treat
out-of-scope items as the "should abstain" class and answerable items as the
"should answer" class, sweep the threshold, and report the value that best
separates them by balanced accuracy.

    python tune_threshold.py
"""

from __future__ import annotations

import sys

from regrag import config, goldset
from regrag.retrieve import max_similarity, retrieve

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def collect() -> list[tuple[float, bool, object]]:
    """Return (max_similarity, should_abstain, item) for every gold item."""
    rows = []
    for item in goldset.load():
        hits = retrieve(item.question)
        rows.append((max_similarity(hits), item.is_out_of_scope, item))
    return rows


def evaluate(rows, threshold: float) -> tuple[float, float, float]:
    answerable = [r for r in rows if not r[1]]
    out_scope = [r for r in rows if r[1]]
    answer_ok = sum(1 for s, _, _ in answerable if s >= threshold) / len(answerable) if answerable else 0.0
    abstain_ok = sum(1 for s, _, _ in out_scope if s < threshold) / len(out_scope) if out_scope else 0.0
    return answer_ok, abstain_ok, (answer_ok + abstain_ok) / 2


def main() -> None:
    rows = collect()
    ans = [s for s, o, _ in rows if not o]
    out = [s for s, o, _ in rows if o]
    print(f"answerable max-similarity:   min {min(ans):.3f}  mean {sum(ans) / len(ans):.3f}")
    print(f"out-of-scope max-similarity: max {max(out):.3f}  mean {sum(out) / len(out):.3f}")

    print(f"\n{'thr':>5} {'answer_ok':>10} {'abstain_ok':>11} {'balanced':>9}")
    best = None
    for i in range(30, 91, 5):
        t = i / 100
        answer_ok, abstain_ok, balanced = evaluate(rows, t)
        print(f"{t:5.2f} {answer_ok:10.2f} {abstain_ok:11.2f} {balanced:9.3f}")
        if best is None or balanced > best[1]:
            best = (t, balanced)

    print(f"\ncurrent config threshold: {config.ABSTAIN_SIMILARITY_THRESHOLD}")
    print(f"best threshold (coarse sweep): {best[0]:.2f}  (balanced accuracy {best[1]:.3f})")

    print("\nmisclassified at the best threshold:")
    any_wrong = False
    for s, should_abstain, item in rows:
        predicts_abstain = s < best[0]
        if predicts_abstain != should_abstain:
            any_wrong = True
            kind = "should ABSTAIN, would answer" if should_abstain else "should ANSWER, would abstain"
            print(f"  {item.id}  sim={s:.3f}  {kind}: {item.question[:55]}")
    if not any_wrong:
        print("  none")


if __name__ == "__main__":
    main()
