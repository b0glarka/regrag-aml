# RegRAG-AML

A small, deployed, and evaluated retrieval-augmented Q&A assistant over a bounded public regulatory corpus: the FATF (Financial Action Task Force) AML/CFT recommendations. It answers questions with inline citations to the source text, abstains when the answer is not in the documents, treats retrieved text as data rather than instructions, and reports real retrieval and answer-quality metrics.

> Live demo: _(link added after deploy)_

## Why this project

Entry and mid-level AI/ML roles ask for shipped RAG systems, evaluation frameworks for generative outputs, and a live deploy. RegRAG-AML is a deliberately small build that produces exactly those three, over a compliance corpus that also fits a financial-crime / regulatory wedge. The faithfulness evaluation and injection-aware guardrails reuse methodology from an MS Business Analytics capstone on prompt-injection defenses.

## Problem / motivation

_(To be written: why grounded, cited, abstaining answers matter for regulatory text, and why a prototype-only demo is not enough.)_

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

- Corpus: FATF "The FATF Recommendations" (the 40 Recommendations). One corpus, one embedding model, one vector store.
- Embeddings: local `sentence-transformers` (no API cost, fully offline for eval).
- Vector store: Chroma (persisted locally).
- Generation: Together AI (Llama 3.3 70B Instruct Turbo) by default, behind a provider-agnostic client that also supports OpenRouter and OpenAI-compatible endpoints.
- Faithfulness judge: Anthropic Haiku 4.5 with a versioned rubric.

## Evaluation

_(Results table added on Day 2. Reports retrieval quality — recall@k, MRR — and answer quality — citation-correctness / faithfulness via an LLM judge validated against a small human-labeled subset. Numbers are indicative on a hand-written gold set, not a benchmark.)_

## How to run locally

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env   # then fill in the keys you need

# 3. Download the corpus (see scripts/, added on Day 1)
# 4. Build the index, then launch the app (commands added as the pipeline lands)
```

## Limitations and next steps

_(To be written honestly: small hand-written gold set, single corpus, no reranking, single embedding model, judge agreement caveats, and what a production version would add.)_

## License

MIT. See [LICENSE](LICENSE).
