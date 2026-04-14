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


from app.embedding.sparse_embedder import SparseEmbedder

class Retriever:

    def __init__(
        self,
        embedder: BaseEmbedder,
        sparse_embedder: SparseEmbedder,
        vectorstore: QdrantVectorStore,
    ):
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.vectorstore = vectorstore

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Returns top_k most relevant payloads using hybrid search.
        """
        query_dense = self.embedder.embed_text(query)
        query_sparse = self.sparse_embedder.embed_text(query)
        
        results = self.vectorstore.search(
            query_dense=query_dense, 
            query_sparse=query_sparse, 
            top_k=top_k
        )

        payloads = []

        for r in results:
            if r.payload:
                payload = dict(r.payload)
                payload["score"] = r.score
                payloads.append(payload)

        return payloads
