"""
app/streamlit_app.py
RAG Document Q&A System — Streamlit UI (EN / AZ)
Supports PDF upload and URL input.
"""

import streamlit as st
import tempfile
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.document_loader import load_pdf, load_url, chunk_text
from src.retrieval.vector_store import VectorStore
from src.generation.rag_chain import SimpleRAGChain

# ── Language strings ─────────────────────────────────────────────
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
        "processing": "Processing documents...",
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
        "processing": "Emal edilir...",
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
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
if "source_count" not in st.session_state:
    st.session_state.source_count = 0

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

    # ── PDF tab ──
    with pdf_tab:
        uploaded_files = st.file_uploader(
            T["uploader"],
            type=["pdf"],
            accept_multiple_files=True
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                pages = load_pdf(tmp_path)
                chunks = chunk_text(pages)
                for chunk in chunks:
                    chunk.source = uploaded_file.name
                # Avoid duplicates
                existing_sources = {c.source for c in st.session_state.all_chunks}
                if uploaded_file.name not in existing_sources:
                    st.session_state.all_chunks.extend(chunks)
                    st.session_state.source_count += 1
                os.unlink(tmp_path)

    # ── URL tab ──
    with url_tab:
        url_input = st.text_input(T["url_input"], placeholder="https://example.com")
        if st.button(T["url_btn"]) and url_input:
            with st.spinner(T["url_loading"]):
                pages = load_url(url_input)
                if pages:
                    chunks = chunk_text(pages)
                    existing_sources = {c.source for c in st.session_state.all_chunks}
                    if url_input not in existing_sources:
                        st.session_state.all_chunks.extend(chunks)
                        st.session_state.source_count += 1
                    st.success(f"✅ {len(chunks)} {T['url_success']}")
                else:
                    st.error(T["url_error"])

    st.divider()
    top_k = st.slider(T["slider"], min_value=1, max_value=10, value=5)

    if st.session_state.all_chunks and st.button(T["build_btn"], type="primary"):
        with st.spinner(T["indexing"]):
            store = VectorStore()
            store.build(st.session_state.all_chunks)
            st.session_state.vector_store = store
        st.success(f"✅ {len(st.session_state.all_chunks)} {T['success']}")
        st.metric(T["metric_chunks"], len(st.session_state.all_chunks))
        st.metric(T["metric_docs"], st.session_state.source_count)

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
