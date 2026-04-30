"""
app.py – Streamlit web UI for the College Student Knowledge Assistant.

Run with:  streamlit run app.py
"""

import os
import sys
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after path setup) ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import DOCS_DIR
from retrieval.vector_store import VectorStore
from retrieval.search import search
from generation.rag import ask_with_history
from ingest.pipeline import ingest_folder, ingest_document


# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []      # chat messages for display
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []  # Anthropic-format history for RAG
if "store" not in st.session_state:
    st.session_state.store = VectorStore()


store = st.session_state.store

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=64)
    st.title("Knowledge Assistant")
    st.caption("Powered by RAG + Claude")

    st.divider()

    # ── API Key ───────────────────────────────────────────────────────────────
    st.subheader("🔑 API Key")
    api_key = st.text_input("Anthropic API Key",
                             type="password",
                             value=os.environ.get("ANTHROPIC_API_KEY", ""),
                             placeholder="sk-ant-...")
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()

    # ── Ingest Documents ──────────────────────────────────────────────────────
    st.subheader("📚 Ingest Documents")

    ingest_mode = st.radio("Source", ["Sample Docs", "Upload File", "Folder Path"])

    if ingest_mode == "Sample Docs":
        if st.button("📂 Load Sample Documents", use_container_width=True):
            with st.spinner("Ingesting sample documents…"):
                results = ingest_folder(str(DOCS_DIR), store=store)
            st.success(f"Loaded {sum(results.values())} chunks from {len(results)} files!")

    elif ingest_mode == "Upload File":
        uploaded = st.file_uploader("Upload a document", type=["txt", "pdf", "docx", "html"])
        doc_type = st.text_input("doc_type tag", placeholder="e.g. handbook")
        if uploaded and st.button("Ingest Uploaded File", use_container_width=True):
            import tempfile, pathlib
            suffix = pathlib.Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            with st.spinner("Ingesting…"):
                n = ingest_document(tmp_path,
                                    extra_meta={"source": uploaded.name,
                                                "doc_type": doc_type or "uploaded"},
                                    store=store)
            st.success(f"Ingested {n} chunks from '{uploaded.name}'")

    else:  # Folder Path
        folder_path = st.text_input("Folder path", placeholder="/path/to/docs")
        if folder_path and st.button("Ingest Folder", use_container_width=True):
            with st.spinner("Ingesting folder…"):
                results = ingest_folder(folder_path, store=store)
            st.success(f"Loaded {sum(results.values())} chunks from {len(results)} files!")

    st.divider()

    # ── Search Settings ───────────────────────────────────────────────────────
    st.subheader("⚙️ Search Settings")
    search_mode = st.selectbox("Search Mode", ["hybrid", "semantic", "keyword"],
                                help="hybrid = best of both; semantic = meaning-based; keyword = exact terms")
    top_k = st.slider("Top-K Chunks", 1, 10, 5)

    # Metadata filters
    st.subheader("🔍 Metadata Filters")
    filter_doc_type = st.text_input("Filter by doc_type", placeholder="e.g. handbook")
    filter_year     = st.text_input("Filter by year",     placeholder="e.g. 2025")

    filters: dict = {}
    if filter_doc_type:
        filters["doc_type"] = filter_doc_type
    if filter_year:
        filters["year"] = filter_year

    st.divider()

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.subheader("📊 Index Stats")
    chunk_count = store.count()
    st.metric("Chunks Indexed", chunk_count)
    if chunk_count > 0:
        sources = store.list_sources()
        st.caption("**Sources:**")
        for src in sources:
            st.caption(f"• {src}")

    if st.button("🗑️ Clear Index", use_container_width=True):
        store.clear()
        st.session_state.history    = []
        st.session_state.rag_history = []
        st.warning("Index cleared.")

    if st.button("🔄 Clear Chat", use_container_width=True):
        st.session_state.history    = []
        st.session_state.rag_history = []
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
st.title("🎓 College Student Knowledge Assistant")
st.caption("Ask anything about your college – attendance rules, exams, placements, scholarships, and more.")

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.history:
    role  = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        if role == "assistant" and "sources" in msg:
            _show_sources(msg["sources"]) if "sources" in msg else None

# Forward-declare helper so it's available in the loop above
def _show_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander("📄 Sources cited", expanded=False):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**{i}. {src['source']}** (relevance: {src['score']:.2f})")
            st.caption(f"> {src['snippet']} …")
            st.divider()

# Rebuild history display using the _show_sources function
def _render_history():
    for msg in st.session_state.history:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])
            if role == "assistant" and msg.get("sources"):
                _show_sources(msg["sources"])

_render_history()

# ── Query Bar ─────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question about your college… (e.g. What is the minimum attendance required?)")

if query:
    # Check prerequisites
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("⚠️ Please enter your Anthropic API key in the sidebar.")
        st.stop()
    if store.count() == 0:
        st.warning("⚠️ No documents indexed yet. Click 'Load Sample Documents' in the sidebar first.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    # Stream assistant response
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        sources_placeholder = st.empty()

        with st.spinner("🔍 Searching documents…"):
            result = ask_with_history(
                query       = query,
                history     = st.session_state.rag_history,
                mode        = search_mode,
                top_k       = top_k,
                filters     = filters if filters else None,
                store       = store,
            )

        answer_placeholder.markdown(result["answer"])
        _show_sources(result["sources"])

    # Update session state
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
    st.session_state.rag_history = result["history"]
    st.rerun()
