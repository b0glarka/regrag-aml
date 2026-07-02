"""Evaluation metrics for RegRAG.

Retrieval:      recall@k (a relevant page in the top-k), reciprocal rank (MRR).
Citations:      citation recall / precision vs the expected pages.
Answer overlap: token-F1 and exact match vs the reference answer. These are
                weak lexical proxies (they penalize paraphrase and reward
                verbosity); the primary answer metric is the LLM judge, not
                these. EM in particular is near-useless for generative answers
                and is reported only as a footnote.
"""

from __future__ import annotations

import re
from collections import Counter

# --- Retrieval -------------------------------------------------------------


def recall_at_k(expected_pages, retrieved_pages, k: int) -> float:
    """1.0 if any expected page appears in the top-k retrieved pages, else 0.0."""
    if not expected_pages:
        return 0.0
    top = list(retrieved_pages)[:k]
    return 1.0 if any(p in top for p in expected_pages) else 0.0


def reciprocal_rank(expected_pages, retrieved_pages) -> float:
    """1/rank of the first retrieved page that is expected; 0.0 if none."""
    expected = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected:
            return 1.0 / rank
    return 0.0


# --- Citations -------------------------------------------------------------

_CITE_RE = re.compile(r"\[pp?\.?\s*([0-9,\s\-–]+)\]", re.IGNORECASE)


def parse_cited_pages(text: str) -> set[int]:
    """Extract page numbers from inline citations like [p. 15] or [pp. 15-16]."""
    pages: set[int] = set()
    for match in _CITE_RE.finditer(text):
        for num in re.findall(r"\d+", match.group(1)):
            pages.add(int(num))
    return pages


def citation_recall(expected_pages, cited_pages) -> float:
    """Fraction of expected pages that the answer actually cited."""
    expected = set(expected_pages)
    if not expected:
        return 0.0
    return len(expected & set(cited_pages)) / len(expected)


def citation_precision(expected_pages, cited_pages) -> float:
    """Fraction of cited pages that were expected. 0.0 if nothing was cited."""
    cited = set(cited_pages)
    if not cited:
        return 0.0
    return len(cited & set(expected_pages)) / len(cited)


# --- Answer overlap (secondary; lexical only) ------------------------------


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token-overlap F1. Weak proxy; report with caveats."""
    pred = _normalize(prediction)
    ref = _normalize(reference)
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred)
    recall = same / len(ref)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """Normalized exact match. Near-zero for generative answers; a footnote."""
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


if __name__ == "__main__":
    exp = [14, 15]
    retr = [67, 14, 22, 15, 3]
    print("recall@5:", recall_at_k(exp, retr, 5))       # 1.0
    print("MRR:", round(reciprocal_rank(exp, retr), 3)) # 0.5 (first hit at rank 2)
    ans = "Financial institutions must undertake CDD [p. 14] and keep records [p. 15]."
    cites = parse_cited_pages(ans)
    print("cited:", sorted(cites))                      # [14, 15]
    print("cite recall:", citation_recall(exp, cites))  # 1.0
    print("cite precision:", citation_precision(exp, cites))  # 1.0
    print("F1:", round(token_f1("keep records five years", "records must be kept for five years"), 3))
    print("EM:", exact_match("five years", "five years"))
