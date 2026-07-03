# RegRAG-AML evaluation

- Embedding: `BAAI/bge-small-en-v1.5`
- Generator: `openrouter:meta-llama/llama-3.3-70b-instruct`
- Judge: `claude-haiku-4-5` (rubric v1.0)
- Retrieval k = 8, abstain threshold = 0.66
- Gold set: 37 items (32 answerable, 5 out-of-scope)

## Headline metrics

| Metric | Value |
|---|---|
| Retrieval recall@8 | 0.969 |
| Retrieval MRR | 0.644 |
| Citation hit rate (cites a correct page) | 0.75 |
| Citation recall (strict, all pages) | 0.57 |
| Citation precision | 0.365 |
| Answer faithfulness (LLM judge) | 0.828 |
| Out-of-scope handled (no fabrication) | 0.9 |
| False-abstention rate (answerable) | 0.0 |
| Lexical token-F1 (secondary) | 0.139 |
| Exact match (footnote, ~0 expected) | 0.0 |

Faithfulness (the LLM judge) is the primary answer-quality metric. Citation hit rate (does the answer cite at least one correct page) is the primary citation metric; strict recall is shown too but is harsh when a Recommendation spans several pages. Token-F1 is a secondary lexical-overlap proxy that penalizes paraphrase; exact match is near zero for generative answers by design and is not a quality signal.

## By difficulty

| Difficulty | N | recall@k | citation recall | faithfulness |
|---|---|---|---|---|
| easy | 10 | 1.0 | 0.583 | 0.95 |
| medium | 18 | 0.944 | 0.597 | 0.806 |
| hard | 4 | 1.0 | 0.416 | 0.625 |

## Coverage

```
Recommendation coverage: 23/40 touched by the gold set.

Covered: R.1(6), R.10(4), R.11(1), R.12(1), R.13(1), R.14(1), R.15(3), R.16(2), R.20(1), R.22(1), R.24(2), R.25(2), R.26(3), R.27(3), R.28(3), R.29(3), R.30(1), R.31(2), R.32(1), R.33(1), R.37(1), R.38(1), R.40(1)

Gaps: R.2, R.3, R.4, R.5, R.6, R.7, R.8, R.9, R.17, R.18, R.19, R.21, R.23, R.34, R.35, R.36, R.39
```

![coverage heatmap](reports/figures/coverage_heatmap.png)
