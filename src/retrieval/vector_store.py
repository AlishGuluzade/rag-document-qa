"""
src/retrieval/vector_store.py
Converts chunks into vectors and stores them in a FAISS index.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from src.ingestion.document_loader import DocumentChunk


# Supports Azerbaijani, Turkish, and English
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

INDEX_PATH = "data/processed/faiss_index.bin"
CHUNKS_PATH = "data/processed/chunks.pkl"


class VectorStore:
    def __init__(self, model_name: str = EMBED_MODEL):
        print(f"[+] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks: List[DocumentChunk] = []

    def build(self, chunks: List[DocumentChunk]):
        """Embeds chunks and builds the FAISS index."""
        print(f"[+] Embedding {len(chunks)} chunks...")
        texts = [chunk.text for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.chunks = chunks

        print(f"[OK] Index built: {self.index.ntotal} vectors, dim={dimension}")

    def save(self):
        """Saves the index to disk."""
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[OK] Index saved: {INDEX_PATH}")

    def load(self):
        """Loads a saved index from disk."""
        self.index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[OK] Index loaded: {self.index.ntotal} vectors")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Finds the most relevant chunks for a query."""
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(score)))

        return results


if __name__ == "__main__":
    from src.ingestion.document_loader import process_documents

    chunks = process_documents("data/raw")

    if chunks:
        store = VectorStore()
        store.build(chunks)
        store.save()

        results = store.search("what are the contract terms?", top_k=3)
        print("\nSearch results:")
        for chunk, score in results:
            print(f"  [{score:.3f}] {chunk.source} p.{chunk.page}: {chunk.text[:100]}...")
