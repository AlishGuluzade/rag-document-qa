"""
app/streamlit_app.py
RAG Document Q&A System — Streamlit UI (EN / AZ)
Supports PDF upload and URL input with source filtering.
"""

import streamlit as st
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.document_loader import load_pdf, load_url, chunk_text
from src.retrieval.vector_store import VectorStore
from src.generation.rag_chain import SimpleRAGChain

LANG = {
    "en": {
        "title": "📄 RAG Document Q&A System",
        "caption": "Upload PDFs or paste a website URL — AI answers your questions",
        "sidebar_header": "📁 Add Sources",
        "pdf_tab": "📄 PDF Upload",
        "url_tab": "🌐 Website URL",
        "uploader": "Choose PDF files",
        "url_input": "Paste a website URL",
        "url_btn": "🔍 Load URL",
        "url_loading": "Fetching website content...",
        "url_success": "chunks loaded from URL!",
        "url_error": "Could not load URL. Please check the link.",
        "slider": "How many chunks to retrieve?",
        "build_btn": "🔍 Build Index",
        "indexing": "Building vector index...",
        "success": "chunks indexed!",
        "metric_chunks": "Total chunks",
        "metric_docs": "Sources",
        "info": "👈 Add PDFs or a URL from the left panel, then build the index",
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
        "filter_title": "🔎 Search in",
        "filter_all": "All sources",
        "no_source": "Please select at least one source.",
    },
    "az": {
        "title": "📄 RAG Document Q&A Sistemi",
        "caption": "PDF yükləyin və ya sayt linki yapışdırın — AI suallarınızı cavablandırır",
        "sidebar_header": "📁 Mənbə əlavə edin",
        "pdf_tab": "📄 PDF",
        "url_tab": "🌐 Sayt linki",
        "uploader": "PDF faylları seçin",
        "url_input": "Sayt linkini yapışdırın",
        "url_btn": "🔍 Yüklə",
        "url_loading": "Sayt oxunur...",
        "url_success": "parça yükləndi!",
        "url_error": "Link yüklənmədi. Linki yoxlayın.",
        "slider": "Neçə parça axtarılsın?",
        "build_btn": "🔍 İndeks qur",
        "indexing": "Vektor indeksi qurulur...",
        "success": "parça indeksləndi!",
        "metric_chunks": "Cəmi chunk",
        "metric_docs": "Mənbə sayı",
        "info": "👈 Sol paneldən PDF və ya link əlavə edin",
        "sources_label": "mənbə",
        "sources_show": "mənbə göstər",
        "page_label": "Səhifə",
        "relevance": "uyğunluq",
        "chat_input": "Sual verin...",
        "searching": "Cavab axtarılır...",
        "stats_title": "📊 Statistika",
        "metric_vectors": "İndekslənmiş vektor",
        "metric_questions": "Sual sayı",
        "clear_btn": "🗑️ Tarixi təmizlə",
        "filter_title": "🔎 Axtarış yeri",
        "filter_all": "Bütün mənbələr",
        "no_source": "Ən azı bir mənbə seçin.",
    }
}

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "sources" not in st.session_state:
    # {name: [chunks]} dict
    st.session_state.sources = {}
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

    pdf_tab, url_tab = st.tabs([T["pdf_tab"], T["url_tab"]])

    with pdf_tab:
        uploaded_files = st.file_uploader(
            T["uploader"], type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.sources:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    pages = load_pdf(tmp_path)
                    chunks = chunk_text(pages)
                    for chunk in chunks:
                        chunk.source = uploaded_file.name
                    st.session_state.sources[uploaded_file.name] = chunks
                    os.unlink(tmp_path)

    with url_tab:
        url_input = st.text_input(T["url_input"], placeholder="https://example.com")
        if st.button(T["url_btn"]) and url_input:
            if url_input not in st.session_state.sources:
                with st.spinner(T["url_loading"]):
                    pages = load_url(url_input)
                    if pages:
                        chunks = chunk_text(pages)
                        for chunk in chunks:
                            chunk.source = url_input
                        st.session_state.sources[url_input] = chunks
                        st.success(f"✅ {len(chunks)} {T['url_success']}")
                    else:
                        st.error(T["url_error"])

    st.divider()
    top_k = st.slider(T["slider"], min_value=1, max_value=10, value=5)

    if st.session_state.sources and st.button(T["build_btn"], type="primary"):
        all_chunks = []
        for chunks in st.session_state.sources.values():
            all_chunks.extend(chunks)
        with st.spinner(T["indexing"]):
            store = VectorStore()
            store.build(all_chunks)
            st.session_state.vector_store = store
            st.session_state.all_chunks = all_chunks
        st.success(f"✅ {len(all_chunks)} {T['success']}")

# ── Main panel ───────────────────────────────────────────────────
T = LANG[st.session_state.lang]
st.title(T["title"])
st.caption(T["caption"])

col1, col2 = st.columns([2, 1])

with col1:
    if st.session_state.vector_store is None:
        st.info(T["info"])
    else:
        # ── Source filter ────────────────────────────────────────
        st.subheader(T["filter_title"])
        source_names = list(st.session_state.sources.keys())

        selected_sources = []
        cols = st.columns(min(len(source_names), 3))
        for i, name in enumerate(source_names):
            short = name if len(name) < 30 else name[:27] + "..."
            if cols[i % 3].checkbox(short, value=True, key=f"src_{name}"):
                selected_sources.append(name)

        st.divider()

        # ── Chat input on top ────────────────────────────────────
        query = st.chat_input(T["chat_input"])
        if query:
            if not selected_sources:
                st.warning(T["no_source"])
            else:
                with st.spinner(T["searching"]):
                    # Filter chunks by selected sources
                    filtered_chunks = []
                    for name in selected_sources:
                        filtered_chunks.extend(st.session_state.sources[name])

                    # Build temp index from selected sources only
                    temp_store = VectorStore()
                    temp_store.build(filtered_chunks)
                    results = temp_store.search(query, top_k=top_k)
                    response = st.session_state.rag_chain.answer(
                        query, results, lang=st.session_state.lang
                    )
                st.session_state.chat_history.append(response)
                st.rerun()

        # Chat history below input
        for item in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(item["query"])
            with st.chat_message("assistant"):
                st.write(item["answer"])
                with st.expander(f"📚 {len(item['sources'])} sources"):
                    for src in item["sources"]:
                        st.markdown(f"**{src['file']}** — Page {src['page']} (relevance: {src['relevance']})")
                        st.caption(src.get("preview", ""))

with col2:
    if st.session_state.vector_store:
        st.subheader(T["stats_title"])
        st.metric(T["metric_vectors"], st.session_state.vector_store.index.ntotal)
        st.metric(T["metric_questions"], len(st.session_state.chat_history))
        st.metric(T["metric_docs"], len(st.session_state.sources))
        if st.button(T["clear_btn"]):
            st.session_state.chat_history = []
            st.rerun()
