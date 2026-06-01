# 📄 RAG Document Q&A System

A multilingual (Azerbaijani / Turkish / English) document question-answering system powered by Retrieval-Augmented Generation (RAG).

> Upload any PDF → Ask questions → Get answers with source references

## 🚀 Live Demo

*Coming soon — HuggingFace Spaces*

## 🏗️ Architecture

```
PDF Document
     ↓
[Document Loader]  →  Pages  →  Chunks (500 words, 50 overlap)
     ↓
[Embedding Model]  →  paraphrase-multilingual-mpnet-base-v2
     ↓
[FAISS Index]      →  Cosine similarity search
     ↓
[RAG Chain]        →  Context + Query → LLM → Answer + Sources
     ↓
[Streamlit UI]     →  Interactive bilingual chat interface
```

## ✨ Features

- 🌍 **Multilingual** — works with Azerbaijani, Turkish, and English documents
- 📚 **Source-aware** — every answer shows which page it came from
- 🛡️ **Hallucination-reduced** — answers strictly from document context
- 🌐 **Bilingual UI** — switch between Azerbaijani and English interface
- 🆓 **Free deployment** — runs on HuggingFace Spaces

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers (multilingual) |
| Vector store | FAISS |
| LLM | Phi-3-mini / Mistral-7B |
| UI | Streamlit |
| Deployment | HuggingFace Spaces |

## ⚙️ Installation

```bash
git clone https://github.com/AlishGuluzade/rag-document-qa
cd rag-document-qa

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## ▶️ Usage

```bash
# Place your PDFs in data/raw/
cp your_document.pdf data/raw/

# Launch the app
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501`, upload PDFs from the sidebar, build the index, and start asking questions.

## 📁 Project Structure

```
rag-document-qa/
├── data/
│   ├── raw/                        ← place PDFs here
│   └── processed/                  ← FAISS index (auto-generated)
├── src/
│   ├── ingestion/
│   │   └── document_loader.py      ← PDF → chunks
│   ├── retrieval/
│   │   └── vector_store.py         ← embeddings + FAISS search
│   └── generation/
│       └── rag_chain.py            ← prompt engineering + LLM
├── app/
│   └── streamlit_app.py            ← bilingual UI
├── notebooks/                      ← exploration & evaluation
└── requirements.txt
```

## 🗺️ Roadmap

- [ ] Evaluation pipeline (RAGAS)
- [ ] Cross-encoder reranking
- [ ] Multi-document comparison mode
- [ ] HuggingFace Spaces deployment
- [ ] Fine-tuning on Azerbaijani corpus

## 👤 Author

**Alish Guluzade** — [GitHub](https://github.com/AlishGuluzade)
