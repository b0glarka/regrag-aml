"""Streamlit app for RegRAG-AML.

Run from the repo root:
    streamlit run app.py     (or: uv run streamlit run app.py)

A thin UI over regrag.generate.answer(): ask a question, get a grounded answer
with page citations, see the retrieved source chunks, and get an explicit
abstention when the corpus does not cover the question.
"""

from __future__ import annotations

import streamlit as st

from regrag import config, store
from regrag.generate import answer

st.set_page_config(page_title="RegRAG-AML", page_icon="🔎")

st.title("RegRAG-AML")
st.caption(
    "Retrieval-augmented Q&A over the FATF AML/CFT Recommendations. Answers are "
    "grounded in the source text with page citations, and the assistant abstains "
    "when the documents do not cover a question."
)

# The vector index must exist (.chroma/ is gitignored; see README).
if store.get_collection().count() == 0:
    st.error(
        "The vector index is empty. From the repo root, run "
        "`python download_corpus.py` then `python build_index.py`, then reload."
    )
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
