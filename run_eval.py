"""Run the full evaluation over the gold set and write a results report.

For each gold item this generates an answer, computes retrieval and citation
metrics, and judges faithfulness with the LLM judge. Answerable items measure
retrieval / citation / faithfulness; out-of-scope items measure whether the
system declined without fabricating (via the same faithfulness judge: a refusal
makes no unsupported claim, so it grades as faithful).

Costs money: about one generation call per item plus one judge call per item
that was answered. Generation is cached under cache/ so re-runs (e.g. after a
rubric change) do not re-spend on generation.

    python run_eval.py            # use the cache where available
    python run_eval.py --refresh  # ignore the cache and regenerate
"""

from __future__ import annotations

import json
import sys
from statistics import mean

from tqdm import tqdm

from regrag import config, goldset, metrics
from regrag import judge as judge_mod
from regrag.coverage import coverage_table, make_heatmap
from regrag.generate import answer

CACHE_PATH = config.REPO_ROOT / "cache" / "eval_answers.json"
JUDGE_CACHE_PATH = config.REPO_ROOT / "cache" / "eval_judgments.json"
RESULTS_PATH = config.REPO_ROOT / "results" / "eval_results.json"
REPORT_PATH = config.REPO_ROOT / "reports" / "eval_report.md"


def _load_cache(refresh: bool) -> dict:
    if refresh or not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate(question: str, cache: dict) -> dict:
    if question in cache:
        return cache[question]
    res = answer(question)
    cache[question] = {
        "answer": res.answer,
        "abstained": res.abstained,
        "max_similarity": res.max_similarity,
        "hits": [
            {"id": h["id"], "page": h["page"], "similarity": h["similarity"], "text": h["text"]}
            for h in res.hits
        ],
    }
    _save_cache(cache)
    return cache[question]


def _load_judge_cache(refresh: bool) -> dict:
    if refresh or not JUDGE_CACHE_PATH.exists():
        return {}
    return json.loads(JUDGE_CACHE_PATH.read_text(encoding="utf-8"))


def _save_judge_cache(cache: dict) -> None:
    JUDGE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    JUDGE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _judge(question: str, answer: str, sources: list[dict], jcache: dict):
    """Judge with caching, keyed by rubric version + question + answer."""
    key = f"{judge_mod.RUBRIC_VERSION}::{question}::{answer}"
    if key in jcache:
        d = jcache[key]
        return judge_mod.Verdict(d["label"], d["score"], d["rationale"])
    v = judge_mod.judge(question, answer, sources)
    jcache[key] = {"label": v.label, "score": v.score, "rationale": v.rationale}
    _save_judge_cache(jcache)
    return v


def _avg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(vals), 3) if vals else None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    refresh = "--refresh" in sys.argv
    items = goldset.load()
    cache = _load_cache(refresh)
    jcache = _load_judge_cache(refresh)

    per_item = []
    for item in tqdm(items, desc="eval"):
        c = _generate(item.question, cache)
        retrieved_pages = [h["page"] for h in c["hits"]]
        cited = sorted(metrics.parse_cited_pages(c["answer"]))
        row = {
            "id": item.id,
            "type": item.type,
            "difficulty": item.difficulty,
            "recommendation": item.recommendation,
            "abstained": c["abstained"],
            "max_similarity": round(c["max_similarity"], 3),
            "cited_pages": cited,
            "expected_pages": item.expected_pages,
        }

        if item.expects_answer:
            row["recall@k"] = metrics.recall_at_k(item.expected_pages, retrieved_pages, config.TOP_K)
            row["mrr"] = round(metrics.reciprocal_rank(item.expected_pages, retrieved_pages), 3)
            row["citation_hit"] = metrics.citation_hit(item.expected_pages, cited)
            row["citation_recall"] = round(metrics.citation_recall(item.expected_pages, cited), 3)
            row["citation_precision"] = round(metrics.citation_precision(item.expected_pages, cited), 3)
            row["f1"] = round(metrics.token_f1(c["answer"], item.reference_answer), 3)
            row["em"] = metrics.exact_match(c["answer"], item.reference_answer)
            row["false_abstention"] = 1.0 if c["abstained"] else 0.0
            if not c["abstained"]:
                v = _judge(item.question, c["answer"], c["hits"], jcache)
                row["faithful_label"] = v.label
                row["faithful_score"] = v.score
                row["faithful_rationale"] = v.rationale
            else:
                row["faithful_label"] = "gate_abstained"
                row["faithful_score"] = None
        else:  # out_of_scope: correct handling = declined without fabricating
            if c["abstained"]:
                row["handled"] = 1.0
                row["faithful_label"] = "gate_abstained"
            else:
                v = _judge(item.question, c["answer"], c["hits"], jcache)
                row["faithful_label"] = v.label
                row["faithful_score"] = v.score
                row["faithful_rationale"] = v.rationale
                row["handled"] = v.score  # faithful=1.0, partial=0.5, unfaithful=0.0

        per_item.append(row)

    _save_cache(cache)
    _save_judge_cache(jcache)

    ans_rows = [r for r in per_item if r["type"] in ("in_scope", "refutation")]
    out_rows = [r for r in per_item if r["type"] == "out_of_scope"]
    judged = [r for r in ans_rows if r.get("faithful_score") is not None]

    aggregates = {
        "n_items": len(items),
        "n_answerable": len(ans_rows),
        "n_out_of_scope": len(out_rows),
        "k": config.TOP_K,
        "abstain_threshold": config.ABSTAIN_SIMILARITY_THRESHOLD,
        "embedding_model": config.EMBEDDING_MODEL,
        "generator": f"{config.GENERATOR_PROVIDER}:{config.GENERATOR_MODEL}",
        "judge_model": config.JUDGE_MODEL,
        "judge_rubric_version": judge_mod.RUBRIC_VERSION,
        "retrieval": {"recall@k": _avg(ans_rows, "recall@k"), "mrr": _avg(ans_rows, "mrr")},
        "citations": {
            "hit_rate": _avg(ans_rows, "citation_hit"),
            "recall": _avg(ans_rows, "citation_recall"),
            "precision": _avg(ans_rows, "citation_precision"),
        },
        "answer": {
            "faithfulness": _avg(judged, "faithful_score"),
            "f1_lexical": _avg(ans_rows, "f1"),
            "em_lexical": _avg(ans_rows, "em"),
            "false_abstention_rate": _avg(ans_rows, "false_abstention"),
        },
        "abstention": {"out_of_scope_handled": _avg(out_rows, "handled")},
    }

    by_diff = {}
    for d in ("easy", "medium", "hard"):
        rows = [r for r in ans_rows if r["difficulty"] == d]
        jrows = [r for r in rows if r.get("faithful_score") is not None]
        by_diff[d] = {
            "n": len(rows),
            "recall@k": _avg(rows, "recall@k"),
            "citation_recall": _avg(rows, "citation_recall"),
            "faithfulness": _avg(jrows, "faithful_score"),
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"aggregates": aggregates, "by_difficulty": by_diff, "items": per_item},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        heatmap = make_heatmap()
    except Exception as exc:  # e.g. matplotlib not installed
        heatmap = None
        print(f"(heatmap skipped: {exc})")
    _write_report(aggregates, by_diff, heatmap)

    print("\nwrote", RESULTS_PATH)
    print("wrote", REPORT_PATH)
    if heatmap:
        print("wrote", heatmap)
    print("\n" + json.dumps(aggregates, indent=2))


def _write_report(agg, by_diff, heatmap) -> None:
    a = agg
    L = [
        "# RegRAG-AML evaluation",
        "",
        f"- Embedding: `{a['embedding_model']}`",
        f"- Generator: `{a['generator']}`",
        f"- Judge: `{a['judge_model']}` (rubric {a['judge_rubric_version']})",
        f"- Retrieval k = {a['k']}, abstain threshold = {a['abstain_threshold']}",
        f"- Gold set: {a['n_items']} items ({a['n_answerable']} answerable, {a['n_out_of_scope']} out-of-scope)",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval recall@{a['k']} | {a['retrieval']['recall@k']} |",
        f"| Retrieval MRR | {a['retrieval']['mrr']} |",
        f"| Citation hit rate (cites a correct page) | {a['citations']['hit_rate']} |",
        f"| Citation recall (strict, all pages) | {a['citations']['recall']} |",
        f"| Citation precision | {a['citations']['precision']} |",
        f"| Answer faithfulness (LLM judge) | {a['answer']['faithfulness']} |",
        f"| Out-of-scope handled (no fabrication) | {a['abstention']['out_of_scope_handled']} |",
        f"| False-abstention rate (answerable) | {a['answer']['false_abstention_rate']} |",
        f"| Lexical token-F1 (secondary) | {a['answer']['f1_lexical']} |",
        f"| Exact match (footnote, ~0 expected) | {a['answer']['em_lexical']} |",
        "",
        "Faithfulness (the LLM judge) is the primary answer-quality metric. Citation hit rate "
        "(does the answer cite at least one correct page) is the primary citation metric; strict "
        "recall is shown too but is harsh when a Recommendation spans several pages. Token-F1 is a "
        "secondary lexical-overlap proxy that penalizes paraphrase; exact match is near zero for "
        "generative answers by design and is not a quality signal.",
        "",
        "## By difficulty",
        "",
        "| Difficulty | N | recall@k | citation recall | faithfulness |",
        "|---|---|---|---|---|",
    ]
    for d in ("easy", "medium", "hard"):
        b = by_diff[d]
        L.append(f"| {d} | {b['n']} | {b['recall@k']} | {b['citation_recall']} | {b['faithfulness']} |")
    L += ["", "## Coverage", "", "```", coverage_table(), "```", ""]
    if heatmap is not None:
        L += [f"![coverage heatmap]({heatmap.relative_to(config.REPO_ROOT).as_posix()})", ""]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
