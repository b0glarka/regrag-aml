# RegRAG-AML

A small, deployed, and evaluated retrieval-augmented Q&A assistant over a bounded public regulatory corpus: the FATF (Financial Action Task Force) AML/CFT recommendations. It answers questions with inline citations to the source text, abstains when the answer is not in the documents, treats retrieved text as data rather than instructions, and reports real retrieval and answer-quality metrics.

> Live demo: _(link added after deploy)_

## Why this project

Entry and mid-level AI/ML roles ask for shipped RAG systems, evaluation frameworks for generative outputs, and a live deploy. RegRAG-AML is a deliberately small build that produces exactly those three, over a compliance corpus that also fits a financial-crime / regulatory wedge. The faithfulness evaluation and injection-aware guardrails reuse methodology from an MS Business Analytics capstone on prompt-injection defenses.

## Problem / motivation

### What is FATF, and why this corpus

The Financial Action Task Force (FATF) is the intergovernmental body, established by the G7 in 1989, that sets the global standard for combating money laundering, terrorist financing, and the financing of proliferation. Its core output, "The FATF Recommendations" (the 40 Recommendations), is the framework that more than 200 jurisdictions have committed to implement. FATF assesses countries against it through mutual evaluations, and jurisdictions with strategic deficiencies can be placed on its "grey" or "black" lists, which carries real economic and reputational consequences. In practice, the Recommendations shape the AML/CFT obligations that banks and other regulated institutions must meet worldwide.

That makes it an ideal corpus for a grounded question-answering assistant: it is authoritative, public, bounded, and well-structured, and it is exactly the kind of high-stakes regulatory text where a wrong or invented answer is unacceptable. A compliance assistant that hallucinates an obligation, or cites a rule that is not there, is worse than useless.

RegRAG-AML therefore does three things a demo chatbot does not: it answers only from the source text with page citations, it abstains when the documents do not cover a question, and it reports measured retrieval and answer-faithfulness metrics.

Official source: [The FATF Recommendations](https://www.fatf-gafi.org/en/publications/fatfrecommendations/documents/fatf-recommendations.html) and [fatf-gafi.org](https://www.fatf-gafi.org/).

## Approach

```
FATF corpus (PDF)
      |
   extract + chunk
      |
   embed (local sentence-transformers)
      |
   vector store (Chroma)
      |
   retrieve top-k
      |
   LLM answer with inline citations   <- abstain if low relevance
      |                                   <- retrieved text treated as data, not instructions
   answer + citations
```

- Corpus: FATF "The FATF Recommendations" (the 40 Recommendations). One corpus, one embedding model, one vector store. The PDF is pulled from a pinned Internet Archive (Wayback Machine) snapshot and verified by SHA-256, so the corpus is reproducible and independent of FATF's live CDN. FATF publishes no data API; a pinned, checksummed archive snapshot is the more durable source for a static standards document.
- Embeddings: local `sentence-transformers` (no API cost, fully offline for eval).
- Vector store: Chroma (persisted locally).
- Generation: an OpenRouter-hosted model (Llama 3.3 70B by default) behind a provider-agnostic client that also targets Together AI and any OpenAI-compatible endpoint. For the eval, OpenRouter routing is pinned so a run cannot silently switch upstream vendors.
- Faithfulness judge: Anthropic Haiku 4.5 with a versioned rubric.

## Evaluation

_(Results table added on Day 2. Reports retrieval quality — recall@k, MRR — and answer quality — citation-correctness / faithfulness via an LLM judge validated against a small human-labeled subset. Numbers are indicative on a hand-written gold set, not a benchmark.)_

## How to run locally

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `requirements.txt` is generated from the lockfile and is what Streamlit Community Cloud installs on deploy.

```bash
# 1. Create the environment from the lockfile
uv sync

# 2. Configure API keys
cp .env.example .env   # then fill in the keys you need (see .env.example)

# 3. Download the corpus (gitignored; documented, reproducible download step)
uv run python download_corpus.py

# 4. Sanity check: extract the corpus text
uv run python -m regrag.ingest

# Further steps (build index, launch app) are added as the pipeline lands.
```

To refresh `requirements.txt` after changing dependencies (the project itself is excluded so the deploy installs only third-party packages):

```bash
uv export --no-hashes --no-dev --no-emit-project -o requirements.txt
```

Prefer plain pip? `python -m venv .venv` then `pip install -r requirements.txt` also works.

## Limitations and next steps

_(To be written honestly: small hand-written gold set, single corpus, no reranking, single embedding model, judge agreement caveats, and what a production version would add.)_

## License

MIT. See [LICENSE](LICENSE).
