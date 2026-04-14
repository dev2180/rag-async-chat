"""
MODULE: app/rag/retriever.py

Responsibility:
    Retrieves relevant documents from vector store.

Must NOT:
    - Call LLM
    - Build prompts
"""

from typing import List, Dict
from app.embedding.base import BaseEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore


class Retriever:

    def __init__(
        self,
        embedder: BaseEmbedder,
        vectorstore: QdrantVectorStore,
    ):
        self.embedder = embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns top_k most relevant payloads.
        """
        query_vector = self.embedder.embed_text(query)
        results = self.vectorstore.search(query_vector, top_k=top_k)

        payloads = []

        for r in results:
            if r.payload:
                payload = dict(r.payload)
                payload["score"] = r.score
                payloads.append(payload)

        return payloads
