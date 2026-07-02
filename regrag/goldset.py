"""Load and validate the gold evaluation set.

Schema per item:
  id                str
  question          str
  type              in_scope | refutation | out_of_scope
                      in_scope    -> answerable from the text
                      refutation  -> false premise, still answerable/correctable from the text
                      out_of_scope-> genuinely not in the document; must abstain
  difficulty        easy | medium | hard        (required unless out_of_scope)
  recommendation    str, e.g. "R.10", "INR.1"   (required unless out_of_scope)
  expected_pages    list[int]  (non-empty unless out_of_scope, where it must be [])
  reference_answer  str        (required unless out_of_scope)
  category, notes   str        (optional, informational)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

from regrag import config
from regrag.ingest import load_pages

VALID_TYPES = {"in_scope", "refutation", "out_of_scope"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}
_REC_RE = re.compile(r"^(R|INR)\.\d+", re.IGNORECASE)


@dataclass
class GoldItem:
    id: str
    question: str
    type: str
    expected_pages: list[int]
    reference_answer: str = ""
    difficulty: str = ""
    recommendation: str = ""
    category: str = ""
    notes: str = ""

    @property
    def expects_answer(self) -> bool:
        """in_scope and refutation both expect a grounded, cited answer."""
        return self.type in ("in_scope", "refutation")

    @property
    def is_out_of_scope(self) -> bool:
        return self.type == "out_of_scope"


def load(path=None) -> list[GoldItem]:
    """Typed load. Assumes the file has already passed validate()."""
    path = path or config.GOLD_SET_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    return [
        GoldItem(
            id=str(r["id"]),
            question=str(r["question"]),
            type=str(r["type"]),
            expected_pages=[int(p) for p in (r.get("expected_pages") or [])],
            reference_answer=(r.get("reference_answer") or "").strip(),
            difficulty=(r.get("difficulty") or "").strip().lower(),
            recommendation=(r.get("recommendation") or "").strip(),
            category=(r.get("category") or "").strip(),
            notes=(r.get("notes") or "").strip(),
        )
        for r in raw
    ]


def corpus_page_numbers() -> set[int]:
    return {p["page"] for p in load_pages()}


def validate(path=None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block the eval; warnings are advisory."""
    path = path or config.GOLD_SET_PATH
    errors: list[str] = []
    warnings: list[str] = []

    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError:
        return ([f"Gold set not found at {path}"], [])
    except yaml.YAMLError as exc:
        return ([f"YAML parse error: {exc}"], [])

    if not isinstance(raw, list) or not raw:
        return (["Gold set must be a non-empty YAML list of items."], [])

    valid_pages = corpus_page_numbers()
    seen_ids: set[str] = set()
    answerable = out_scope = hard = 0

    for i, item in enumerate(raw):
        loc = f"item {i + 1}"
        if not isinstance(item, dict):
            errors.append(f"{loc}: not a mapping.")
            continue

        iid = item.get("id")
        loc = f"item {i + 1} (id={iid!r})"
        if not iid or not isinstance(iid, str):
            errors.append(f"{loc}: missing or non-string 'id'.")
        elif iid in seen_ids:
            errors.append(f"{loc}: duplicate id.")
        else:
            seen_ids.add(iid)

        if not item.get("question"):
            errors.append(f"{loc}: missing 'question'.")

        itype = item.get("type")
        if itype not in VALID_TYPES:
            errors.append(f"{loc}: 'type' must be in_scope | refutation | out_of_scope (got {itype!r}).")
            continue

        pages = item.get("expected_pages") or []
        if not isinstance(pages, list) or any(not isinstance(p, int) for p in pages):
            errors.append(f"{loc}: 'expected_pages' must be a list of integers.")
            pages = []
        reference = (item.get("reference_answer") or "").strip()
        difficulty = (item.get("difficulty") or "").strip().lower()
        recommendation = (item.get("recommendation") or "").strip()

        if itype == "out_of_scope":
            out_scope += 1
            if pages:
                errors.append(f"{loc}: out-of-scope item must have empty expected_pages.")
        else:  # in_scope or refutation -> must be answerable and cited
            answerable += 1
            if difficulty not in VALID_DIFFICULTY:
                errors.append(f"{loc}: 'difficulty' must be easy|medium|hard (got {difficulty!r}).")
            elif difficulty == "hard":
                hard += 1
            if not recommendation:
                errors.append(f"{loc}: needs a 'recommendation' (e.g. R.10).")
            elif not _REC_RE.match(recommendation):
                warnings.append(f"{loc}: recommendation {recommendation!r} isn't shaped like 'R.N' or 'INR.N'.")
            if not pages:
                errors.append(f"{loc}: needs at least one expected page.")
            for p in pages:
                if p not in valid_pages:
                    errors.append(f"{loc}: expected_page {p} is not a valid corpus page.")
            if not reference:
                errors.append(f"{loc}: needs a reference_answer.")

    if answerable < 20:
        warnings.append(f"Only {answerable} answerable items (target ~20-25).")
    if out_scope < 3:
        warnings.append(f"Only {out_scope} out-of-scope items (target ~5-8).")
    if hard == 0:
        warnings.append("No 'hard' items (add a few multi-step synthesis questions).")

    return errors, warnings
