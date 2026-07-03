"""Per-Recommendation coverage of the gold set.

Parses each item's `recommendation` field, which may be a single value ("R.10"),
a compound ("R.10+R.16"), or a range ("R.26-R.31"), into the set of underlying
Recommendation numbers. Interpretive notes (INR.N) are folded into their parent
Recommendation (R.N) for the heatmap. Produces a text table and a matplotlib
heatmap over R.1 to R.40 that makes coverage and gaps explicit.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from regrag import config
from regrag.goldset import load

N_RECS = 40

_RANGE_RE = re.compile(r"(R|INR)\.(\d+)\s*-\s*(?:(?:R|INR)\.)?(\d+)", re.IGNORECASE)
_SINGLE_RE = re.compile(r"(R|INR)\.(\d+)", re.IGNORECASE)


def parse_recommendations(rec: str) -> set[int]:
    """Return the set of Recommendation numbers a label refers to (INR folded to R)."""
    numbers: set[int] = set()
    for token in re.split(r"[+,/]", rec or ""):
        token = token.strip()
        if not token:
            continue
        m = _RANGE_RE.match(token)
        if m:
            start, end = int(m.group(2)), int(m.group(3))
            numbers.update(range(min(start, end), max(start, end) + 1))
            continue
        m = _SINGLE_RE.match(token)
        if m:
            numbers.add(int(m.group(2)))
    return {n for n in numbers if 1 <= n <= N_RECS}


def coverage_counts(path=None) -> Counter:
    """Count how many answerable gold items touch each Recommendation number."""
    counts: Counter = Counter()
    for item in load(path):
        if not item.expects_answer:
            continue
        for n in parse_recommendations(item.recommendation):
            counts[n] += 1
    return counts


def coverage_table(path=None) -> str:
    counts = coverage_counts(path)
    covered = sum(1 for n in range(1, N_RECS + 1) if counts.get(n))
    lines = [f"Recommendation coverage: {covered}/{N_RECS} touched by the gold set.", ""]
    gaps = [f"R.{n}" for n in range(1, N_RECS + 1) if not counts.get(n)]
    hit = [f"R.{n}({counts[n]})" for n in range(1, N_RECS + 1) if counts.get(n)]
    lines.append("Covered: " + ", ".join(hit))
    lines.append("")
    lines.append("Gaps: " + (", ".join(gaps) if gaps else "none"))
    return "\n".join(lines)


def make_heatmap(path=None, out: Path | None = None) -> Path:
    """Write an 8x5 heatmap of gold-set coverage over R.1 to R.40."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    counts = coverage_counts(path)
    rows, cols = 8, 5
    grid = np.zeros((rows, cols))
    for n in range(1, N_RECS + 1):
        r, c = divmod(n - 1, cols)
        grid[r, c] = counts.get(n, 0)

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(1, grid.max()))
    for n in range(1, N_RECS + 1):
        r, c = divmod(n - 1, cols)
        val = int(grid[r, c])
        ax.text(
            c, r, f"R.{n}\n{val}", ha="center", va="center",
            color="white" if val > grid.max() / 2 else "black", fontsize=8,
        )
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Gold-set coverage of the FATF Recommendations\n(count per Recommendation; 0 = gap)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="gold questions")
    fig.tight_layout()

    out = out or (config.REPO_ROOT / "reports" / "figures" / "coverage_heatmap.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(coverage_table())
    print("\nHeatmap:", make_heatmap())
