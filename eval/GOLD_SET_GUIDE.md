# Authoring the gold set

The gold set is the small, hand-written test set that the Day 2 evaluation runs against. It drives three measurements:

- Retrieval quality: for each in-scope question, did retrieval surface a page that actually contains the answer (recall@k), and how high did it rank (MRR)?
- Abstention: for out-of-scope questions, did the assistant correctly refuse instead of inventing an answer?
- Answer faithfulness: is the generated answer grounded in the retrieved text, judged against your reference answer?

Aim for quality over quantity. Twenty to thirty careful items is the target from the project scope; a small, correct, well-spread set is worth far more than a large sloppy one.

## File and format

Write the set in `eval/gold_set.yaml` (YAML is far easier to hand-author than JSON: multi-line text, comments, no quote-escaping). Each item is one list entry.

| Field | Required | What it is |
|---|---|---|
| `id` | yes | A short stable id, e.g. `q001`. Never reuse an id. |
| `question` | yes | The question, phrased naturally, as a user would ask it. |
| `in_scope` | yes | `true` for answerable and refutation items; `false` only when the answer is genuinely not in the document and the assistant should abstain. |
| `refutation` | optional | `true` for a false-premise question that is still answerable from the text (the assistant should answer and correct the premise). Refutation items stay `in_scope: true` with real `expected_pages`. |
| `difficulty` | in-scope: yes | `easy`, `medium`, or `hard`. Aim for a spread, including a few `hard` multi-step questions. |
| `recommendation` | in-scope: yes | The source Recommendation, e.g. `R.10` or `INR.1`. Drives the coverage report. |
| `expected_pages` | yes | The printed page number(s) that contain the answer (the numbers in the document footer, which is what `find_page.py` reports). `[]` for out-of-scope items. |
| `reference_answer` | in-scope: yes | A short, correct answer drawn from the text. For out-of-scope items, an optional short "not in the document" note. |
| `category`, `notes` | optional | Informational. |

### Three item types

- Answerable (`in_scope: true`): a normal question the documents answer.
- Refutation (`in_scope: true`, `refutation: true`): a false premise the documents let you correct, for example "does FATF specify a €10,000 threshold?" (no, the threshold is 15,000). The assistant should answer and correct, not abstain, so these are in-scope with real `expected_pages`.
- Out-of-scope (`in_scope: false`): genuinely not in the documents (other frameworks, current events, unrelated topics). The assistant should abstain. Leave `expected_pages` empty.

## How many, and how to spread them

- About 20 to 25 in-scope questions, plus 5 to 8 out-of-scope questions.
- Spread the in-scope ones across the document, not all from one section. Cover a mix of: the numbered Recommendations (R.1 to R.40), the Interpretive Notes, and the Glossary definitions.
- Include a mix of difficulty: simple lookups ("what is required for X"), definition questions ("what is a DNFBP"), and a few that need synthesis across two pages.

## Step-by-step for each question

1. Pick a specific, answerable question. It should have a clear answer in the text, not a yes/no or an opinion.
2. Find the page(s) that contain the answer. Two easy ways:
   - Ask it in the app (`streamlit run app.py`), expand "Show retrieved source chunks", and read the pages it returns. Then confirm the page truly answers the question by checking the PDF.
   - Search the corpus text from the terminal for a phrase you expect (replace the term):

     ```bash
     uv run python -c "from regrag.ingest import load_pages; term='customer due diligence'.lower(); [print(p['page'], '|', p['text'][:90].replace(chr(10),' ')) for p in load_pages() if term in p['text'].lower()]"
     ```

   Record every page that genuinely contains the answer in `expected_pages`. Retrieval counts as correct if any one of them is retrieved, so list all valid pages, but do not pad it with pages that only mention the topic in passing.
3. Write a short reference answer in your own words, faithful to the text. One to three sentences. This is the yardstick the faithfulness judge and you will compare against, so keep it accurate and free of anything not in the source.
4. Fill in `id`, `in_scope: true`, and a `notes` line naming the Recommendation.

## Out-of-scope questions

These test that the assistant refuses rather than hallucinates. Good out-of-scope questions are plausible but genuinely not covered by the FATF Recommendations, for example a question about GDPR data-breach rules, or a specific national law, or a FATF topic that is not in the 40 Recommendations. Set `in_scope: false`, `expected_pages: []`, and leave `reference_answer` empty.

Include one or two that are close to the domain but still not in the text (harder abstention cases), not only obviously unrelated ones.

## Quality checklist

- Every `expected_pages` value has been verified by actually reading that page. This is the most common mistake; do not trust retrieval alone.
- Questions are answerable from the text, not from general knowledge only.
- The reference answer contains nothing that is not supported by the cited page.
- Do not copy a chunk of the PDF verbatim as the question; phrase it as a real user would.
- Variety: not all from the same Recommendation, and a mix of easy and harder items.
- Out-of-scope items are plausible, not just nonsense.

## Later: the faithfulness-labeling subset

After the eval runs and generates answers, you will hand-label a small subset (roughly 15 to 20 answers) as faithful or not faithful to the retrieved sources. That lets us report how well the automated judge agrees with you (Cohen's kappa), the same way your capstone validated its LLM judge. You do not need to do this now; it comes after the answers exist. Just be aware the gold set and that labeling together are what make the eval credible.

## When you are done

Save `eval/gold_set.yaml`, then sanity-check it parses:

```bash
uv run python -c "import yaml; d=yaml.safe_load(open('eval/gold_set.yaml', encoding='utf-8')); print(len(d), 'items'); print('in-scope:', sum(1 for x in d if x['in_scope']))"
```

Tell me when it is ready and I will build the eval harness around it.
