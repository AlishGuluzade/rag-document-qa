"""
app/streamlit_app.py
RAG Document Q&A System — Streamlit UI (EN / AZ)
"""

import streamlit as st
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.document_loader import load_pdf, chunk_text
from src.retrieval.vector_store import VectorStore
from src.generation.rag_chain import SimpleRAGChain

# ── Language strings ─────────────────────────────────────────────
LANG = {
    "en": {
        "page_title": "RAG Document Q&A",
        "title": "📄 RAG Document Q&A System",
        "caption": "Upload PDF documents and ask questions — AI answers",
        "sidebar_header": "📁 Upload Documents",
        "uploader": "Choose PDF files",
        "slider": "How many chunks to retrieve?",
        "build_btn": "🔍 Build Index",
        "processing": "Processing documents...",
        "indexing": "Building vector index...",
        "success": "chunks indexed!",
        "metric_chunks": "Total chunks",
        "metric_docs": "Documents",
        "info": "👈 Upload PDFs from the left panel and build the index",
        "sources_label": "sources",
        "sources_show": "show sources",
        "page_label": "Page",
        "relevance": "relevance",
        "chat_input": "Ask a question about your documents...",
        "searching": "Searching for answer...",
        "stats_title": "📊 Statistics",
        "metric_vectors": "Indexed vectors",
        "metric_questions": "Questions asked",
        "clear_btn": "🗑️ Clear history",
    },
    "az": {
        "page_title": "RAG Sənəd Q&A",
        "title": "📄 RAG Document Q&A Sistemi",
        "caption": "PDF sənədlərinizdən sual soruşun — AI cavab versin",
        "sidebar_header": "📁 Sənəd yükləyin",
        "uploader": "PDF faylları seçin",
        "slider": "Neçə parça axtarılsın?",
        "build_btn": "🔍 İndeks qur",
        "processing": "Sənədlər emal edilir...",
        "indexing": "Vektor indeksi qurulur...",
        "success": "parça indeksləndi!",
        "metric_chunks": "Cəmi chunk",
        "metric_docs": "Sənəd sayı",
        "info": "👈 Sol paneldən PDF yükləyin və indeks qurun",
        "sources_label": "mənbə",
        "sources_show": "mənbə göstər",
        "page_label": "Səhifə",
        "relevance": "uyğunluq",
        "chat_input": "Sənəd haqqında sual verin...",
        "searching": "Cavab axtarılır...",
        "stats_title": "📊 Statistika",
        "metric_vectors": "İndekslənmiş vektor",
        "metric_questions": "Sual sayı",
        "clear_btn": "🗑️ Tarixi təmizlə",
    }
}

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

# ── Session state ────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = SimpleRAGChain()
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    lang_choice = st.selectbox(
        "🌐 Language / Dil",
        options=["en", "az"],
        format_func=lambda x: "🇬🇧 English" if x == "en" else "🇦🇿 Azərbaycan",
        index=0 if st.session_state.lang == "en" else 1,
    )
    if lang_choice != st.session_state.lang:
        st.session_state.lang = lang_choice
        st.rerun()

    T = LANG[st.session_state.lang]

    st.divider()
    st.header(T["sidebar_header"])

    uploaded_files = st.file_uploader(
        T["uploader"],
        type=["pdf"],
        accept_multiple_files=True
    )

    top_k = st.slider(T["slider"], min_value=1, max_value=10, value=5)

    if uploaded_files and st.button(T["build_btn"], type="primary"):
        all_chunks = []
        progress = st.progress(0)

        with st.spinner(T["processing"]):
            for i, uploaded_file in enumerate(uploaded_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                pages = load_pdf(tmp_path)
                chunks = chunk_text(pages)
                for chunk in chunks:
                    chunk.source = uploaded_file.name
                all_chunks.extend(chunks)
                os.unlink(tmp_path)
                progress.progress((i + 1) / len(uploaded_files))

        with st.spinner(T["indexing"]):
            store = VectorStore()
            store.build(all_chunks)
            st.session_state.vector_store = store

        st.success(f"✅ {len(all_chunks)} {T['success']}")
        st.metric(T["metric_chunks"], len(all_chunks))
        st.metric(T["metric_docs"], len(uploaded_files))

# ── Main panel ───────────────────────────────────────────────────
T = LANG[st.session_state.lang]

st.title(T["title"])
st.caption(T["caption"])

col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.vector_store is None:
        st.info(T["info"])
    else:
        for item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(item["query"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                with st.expander(f"📚 {len(item['sources'])} {T['sources_label']}"):
                    for src in item["sources"]:
                        st.markdown(
                            f"**{src['file']}** — {T['page_label']} {src['page']} "
                            f"({T['relevance']}: {src['relevance']})"
                        )
                        st.caption(src.get("preview", ""))

        query = st.chat_input(T["chat_input"])
        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner(T["searching"]):
                    results = st.session_state.vector_store.search(query, top_k=top_k)
                    response = st.session_state.rag_chain.answer(
                        query, results, lang=st.session_state.lang
                    )

                st.write(response["answer"])
                with st.expander(f"📚 {len(response['sources'])} {T['sources_show']}"):
                    for src in response["sources"]:
                        st.markdown(
                            f"**{src['file']}** — {T['page_label']} {src['page']} "
                            f"({T['relevance']}: {src['relevance']})"
                        )
                        if "preview" in src:
                            st.caption(src["preview"])

            st.session_state.chat_history.append(response)

with col2:
    if st.session_state.vector_store:
        st.subheader(T["stats_title"])
        store = st.session_state.vector_store
        st.metric(T["metric_vectors"], store.index.ntotal)
        st.metric(T["metric_questions"], len(st.session_state.chat_history))

        if st.button(T["clear_btn"]):
            st.session_state.chat_history = []
            st.rerun()
