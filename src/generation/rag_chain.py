"""
src/generation/rag_chain.py
RAG pipeline: retrieves context and generates answers using an LLM.
"""

from typing import List, Tuple
from src.ingestion.document_loader import DocumentChunk


def build_prompt(query: str, context_chunks: List[Tuple[DocumentChunk, float]], lang: str = "en") -> str:
    """Builds the RAG prompt from retrieved context chunks."""
    context_text = ""
    for i, (chunk, score) in enumerate(context_chunks, 1):
        context_text += f"\n[{i}. Source: {chunk.source}, Page {chunk.page}]\n{chunk.text}\n"

    if lang == "en":
        prompt = f"""You are an AI assistant that analyzes documents.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "This information was not found in the documents."

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
    else:
        prompt = f"""You are an AI assistant that analyzes documents.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "Bu məlumat sənədlərdə tapılmadı."

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""

    return prompt


class SimpleRAGChain:
    """
    Simple RAG chain for testing — returns the most relevant chunk without an LLM.
    Replace with RAGChain when a full model is available.
    """

    def answer(
        self,
        query: str,
        context_chunks: List[Tuple[DocumentChunk, float]],
        lang: str = "en"
    ) -> dict:
        if not context_chunks:
            msg = (
                "No relevant information found in the documents."
                if lang == "en"
                else "Heç bir uyğun məlumat tapılmadı."
            )
            return {"query": query, "answer": msg, "sources": []}

        best_chunk, best_score = context_chunks[0]

        if lang == "en":
            answer = f"Most relevant information (relevance: {best_score:.2f}):\n\n{best_chunk.text}"
        else:
            answer = f"Ən uyğun məlumat ({best_score:.2f} uyğunluq):\n\n{best_chunk.text}"

        sources = [
            {
                "file": c.source,
                "page": c.page,
                "relevance": round(s, 3),
                "preview": c.text[:120] + "..."
            }
            for c, s in context_chunks
        ]

        return {"query": query, "answer": answer, "sources": sources}
