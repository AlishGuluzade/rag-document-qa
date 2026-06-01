"""
src/ingestion/document_loader.py
Loads PDF documents and splits them into chunks.
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_id: int


def load_pdf(pdf_path: str) -> List[dict]:
    """Reads each page from a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": Path(pdf_path).name
            })
    doc.close()
    return pages


def chunk_text(pages: List[dict], chunk_size: int = 500, overlap: int = 50) -> List[DocumentChunk]:
    """
    Splits text into overlapping chunks.
    chunk_size: number of words per chunk
    overlap: number of words shared between consecutive chunks
    """
    chunks = []
    chunk_id = 0

    for page_data in pages:
        words = page_data["text"].split()
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words)

            chunks.append(DocumentChunk(
                text=chunk_text_str,
                source=page_data["source"],
                page=page_data["page"],
                chunk_id=chunk_id
            ))

            chunk_id += 1
            start += chunk_size - overlap

    return chunks


def process_documents(pdf_folder: str) -> List[DocumentChunk]:
    """Processes all PDFs in the given folder."""
    folder = Path(pdf_folder)
    all_chunks = []

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in {pdf_folder}.")
        return []

    for pdf_path in pdf_files:
        print(f"[+] Loading: {pdf_path.name}")
        pages = load_pdf(str(pdf_path))
        chunks = chunk_text(pages)
        all_chunks.extend(chunks)
        print(f"    -> {len(pages)} pages, {len(chunks)} chunks")

    print(f"\n[OK] Total: {len(all_chunks)} chunks ready")
    return all_chunks


if __name__ == "__main__":
    chunks = process_documents("data/raw")
    if chunks:
        print(f"\nSample chunk:\n{'-'*40}")
        print(f"Source: {chunks[0].source}, Page: {chunks[0].page}")
        print(f"Text: {chunks[0].text[:200]}...")
