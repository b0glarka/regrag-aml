"""LLM-as-judge faithfulness scoring.

A separate judge model (Haiku 4.5) scores whether an answer is faithful to the
retrieved source excerpts: every claim supported, nothing invented. This reuses
the judge-and-validate methodology from the author's capstone. The rubric is
versioned so runs are reproducible and comparable, and so agreement against a
human-labelled subset (Cohen's kappa) can be reported.

The judge is deliberately not the model under test: letting a generator grade
its own output is the failure mode this design avoids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from regrag import config

load_dotenv(config.REPO_ROOT / ".env")

RUBRIC_VERSION = config.JUDGE_RUBRIC_VERSION

_RUBRIC = (
    "You are a strict faithfulness grader for a retrieval-augmented assistant "
    "that answers questions about the FATF Recommendations.\n"
    "You receive a QUESTION, the assistant's ANSWER, and the SOURCE excerpts the "
    "answer was meant to be grounded in.\n"
    "Grade only FAITHFULNESS (grounding), not style, completeness, or whether "
    "you personally know the answer:\n"
    "- faithful: every factual claim in the answer is directly supported by the "
    "sources, and it adds nothing beyond them.\n"
    "- partial: mostly grounded, but with a minor unsupported detail or a slight "
    "overreach.\n"
    "- unfaithful: contains a claim not supported by the sources, contradicts "
    "them, or misattributes.\n"
    "If the ANSWER is an abstention or refusal that makes no factual claim about "
    "the corpus, grade it 'faithful'.\n"
    'Respond with ONLY a JSON object: {"label": "faithful|partial|unfaithful", '
    '"rationale": "one short sentence"}.'
)

_LABEL_SCORE = {"faithful": 1.0, "partial": 0.5, "unfaithful": 0.0}


@dataclass
class Verdict:
    label: str
    score: float
    rationale: str


def _parse(text: str) -> Verdict:
    label, rationale = "unfaithful", ""
    try:
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start : end + 1])
        label = str(obj.get("label", "")).lower().strip()
        rationale = str(obj.get("rationale", "")).strip()
    except Exception:
        low = text.lower()
        if "unfaithful" in low:
            label = "unfaithful"
        elif "partial" in low:
            label = "partial"
        elif "faithful" in low:
            label = "faithful"
    if label not in _LABEL_SCORE:
        label = "unfaithful"
    return Verdict(label=label, score=_LABEL_SCORE[label], rationale=rationale)


def judge(question: str, answer: str, sources: list[dict], model: str | None = None) -> Verdict:
    """Grade one answer's faithfulness against its retrieved sources."""
    model = model or config.JUDGE_MODEL
    src = "\n\n".join(f"[page {s['page']}]\n{s['text']}" for s in sources)
    user = f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSOURCES:\n{src}"

    response = anthropic.Anthropic().messages.create(
        model=model,
        max_tokens=300,
        temperature=0,
        system=_RUBRIC,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return _parse(text)
