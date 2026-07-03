"""Streamlit app for RegRAG-AML.

Run from the repo root:
    streamlit run app.py     (or: uv run streamlit run app.py)

A thin UI over regrag.generate.answer(): ask a question, get a grounded answer
with page citations, see the retrieved source chunks, and get an explicit
abstention when the corpus does not cover the question.
"""

from __future__ import annotations

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

st.set_page_config(page_title="RegRAG-AML", page_icon="🔎")


@st.cache_resource(show_spinner="Preparing the index (first load only)...")
def ensure_index() -> int:
    """Build the vector index if it is missing.

    On the cloud the corpus and store are gitignored, so the app downloads the
    corpus and builds the index on first run. Locally, where the index already
    exists, this is a no-op.
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


st.title("RegRAG-AML")
st.caption(
    "Retrieval-augmented Q&A over the FATF AML/CFT Recommendations. Answers are "
    "grounded in the source text with page citations, and the assistant abstains "
    "when the documents do not cover a question."
)

try:
    ensure_index()
except Exception as exc:
    st.error(f"Could not prepare the vector index: {exc}")
    st.stop()

question = st.text_input(
    "Ask a question about the FATF Recommendations",
    placeholder="e.g. What must financial institutions do for customer due diligence?",
)

if question:
    with st.spinner("Retrieving and answering..."):
        result = answer(question)

    if result.abstained:
        st.warning(result.answer)
        st.caption(
            f"Top similarity {result.max_similarity:.3f} was below the abstain "
            f"threshold {config.ABSTAIN_SIMILARITY_THRESHOLD:.2f}."
        )
    else:
        st.markdown(result.answer)
        pages = ", ".join(f"p. {p}" for p in result.cited_pages)
        st.caption(
            f"Retrieved pages: {pages} · top similarity {result.max_similarity:.3f}"
        )

    with st.expander("Show retrieved source chunks"):
        for i, hit in enumerate(result.hits, start=1):
            st.markdown(
                f"Source {i} — page {hit['page']} (similarity {hit['similarity']:.3f})"
            )
            st.write(hit["text"])
            st.divider()

st.caption(
    "Corpus: FATF Recommendations, from a pinned Internet Archive snapshot. "
    "Portfolio project — not legal or compliance advice."
)
