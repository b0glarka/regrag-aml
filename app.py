"""Streamlit app for RegRAG-AML.

Run from the repo root:
    streamlit run app.py     (or: uv run streamlit run app.py)

A thin UI over regrag.generate.answer(): ask a question (or pick an example),
get a grounded answer with page citations, see the retrieved source chunks, and
get an explicit abstention when the corpus does not cover the question.
"""

from __future__ import annotations

import json
import os

import streamlit as st

from regrag import config, store
from regrag.generate import answer

# Secrets bridge: on Streamlit Community Cloud, API keys arrive via st.secrets,
# not as environment variables. Copy them into the environment so the rest of
# the code (which reads os.getenv) works unchanged. Locally this is a no-op.
try:
    for _key in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

st.set_page_config(page_title="RegRAG-AML", page_icon="🔎", layout="centered")

FATF_DOCS = (
    "https://www.fatf-gafi.org/en/publications/fatfrecommendations/documents/"
    "fatf-recommendations.html"
)
REPO_URL = "https://github.com/b0glarka/regrag-aml"

EXAMPLES = {
    "Customer due diligence": "What must financial institutions do for customer due diligence?",
    "Virtual assets": "How are virtual asset service providers regulated?",
    "Wire transfers": "What information must be included in wire transfer messages?",
    "Out of scope": "Which countries are currently on the FATF grey list?",
}


@st.cache_resource(show_spinner="Preparing the index (first load only)...")
def ensure_index() -> int:
    """Build the vector index if it is missing.

    On the cloud the corpus and store are gitignored, so the app downloads the
    corpus and builds the index on first run. Locally it is a no-op.
    """
    collection = store.get_collection()
    if collection.count() == 0:
        import build_index
        import download_corpus

        if not download_corpus._already_present() and not download_corpus.download():
            raise RuntimeError("corpus download failed")
        build_index.main()
        collection = store.get_collection()
    return collection.count()


def _eval_metrics() -> dict | None:
    try:
        path = config.REPO_ROOT / "results" / "eval_results.json"
        return json.loads(path.read_text(encoding="utf-8"))["aggregates"]
    except Exception:
        return None


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("RegRAG-AML")
    st.write(
        "Grounded question answering over the FATF anti-money-laundering "
        "Recommendations, with page citations and graceful abstention."
    )
    metrics = _eval_metrics()
    if metrics:
        st.subheader("Evaluation")
        st.metric("Retrieval recall@8", metrics["retrieval"]["recall@k"])
        st.metric("Answer faithfulness (LLM judge)", metrics["answer"]["faithfulness"])
        st.metric("Out-of-scope handled", metrics["abstention"]["out_of_scope_handled"])
        st.caption(f"On a hand-written {metrics['n_items']}-item gold set.")
    st.subheader("Links")
    st.markdown(f"- [FATF Recommendations]({FATF_DOCS})\n- [Source code]({REPO_URL})")
    st.caption(
        "Not legal or compliance advice. Answers are drawn only from the "
        "October 2025 FATF Recommendations."
    )

# --- Main ------------------------------------------------------------------
st.title("RegRAG-AML")
st.caption(
    "Ask about the FATF AML/CFT Recommendations. Answers are grounded in the "
    "source text with page citations, and the assistant abstains when the "
    "documents do not cover a question."
)

try:
    ensure_index()
except Exception as exc:
    st.error(f"Could not prepare the vector index: {exc}")
    st.stop()

if "question" not in st.session_state:
    st.session_state["question"] = ""

st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (label, example_q) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state["question"] = example_q

question = st.text_input(
    "Ask a question about the FATF Recommendations",
    key="question",
    placeholder="e.g. What must financial institutions do for customer due diligence?",
)

if question:
    try:
        with st.spinner("Retrieving and answering..."):
            result = answer(question)
    except Exception as exc:
        st.error(f"Sorry, something went wrong answering that: {exc}")
        st.stop()

    if result.abstained:
        st.warning(result.answer)
        st.caption(
            f"Top similarity {result.max_similarity:.3f} was below the abstain "
            f"threshold {config.ABSTAIN_SIMILARITY_THRESHOLD:.2f}."
        )
    else:
        st.markdown(result.answer)
        pages = ", ".join(f"p. {p}" for p in result.cited_pages)
        st.caption(f"Retrieved pages: {pages} · top similarity {result.max_similarity:.3f}")

    with st.expander("Show retrieved source chunks"):
        for i, hit in enumerate(result.hits, start=1):
            st.markdown(f"Source {i} — page {hit['page']} (similarity {hit['similarity']:.3f})")
            st.write(hit["text"])
            st.divider()
