"""
MODULE: app/rag/reranker.py

Implements a Cross-Encoder reranking layer.
Reranks retrieved candidate chunks by scoring their semantic relevance
to the query, improving precision over dual-encoder vector search alone.
"""

import logging
from typing import List, Dict
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading CrossEncoder model '{model_name}'...")
        self.model = CrossEncoder(model_name)
        logger.info("CrossEncoder model loaded successfully.")

    def rerank(self, query: str, payloads: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank a list of payloads based on the query.

        Args:
            query: The search query.
            payloads: List of dictionaries, each containing at least a 'text' key.
            top_k: Number of optimal chunks to keep after reranking.

        Returns:
            Sorted and truncated list of payloads with updated 'score' keys.
        """
        if not payloads:
            return []

        # Create pairs: (query, document_text)
        # Using a list comprehension ensures order matches payloads
        pairs = [[query, p.get("text", "")] for p in payloads]

        # Get scores from the cross encoder
        scores = self.model.predict(pairs)

        # Update scores in payloads
        for i, payload in enumerate(payloads):
            payload["score"] = float(scores[i])

        # Sort descending by the new cross-encoder score
        reranked_payloads = sorted(payloads, key=lambda x: x["score"], reverse=True)

        # Return only the top chunks
        return reranked_payloads[:top_k]
