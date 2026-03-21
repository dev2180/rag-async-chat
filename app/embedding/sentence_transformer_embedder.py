"""
MODULE: app/embedding/sentence_transformer_embedder.py

Sentence-transformers implementation of BaseEmbedder.
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.embedding.base import BaseEmbedder
from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self._dimension}")

    def embed_text(self, text: str) -> List[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        return self._dimension
