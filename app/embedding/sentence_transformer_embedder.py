"""
MODULE: app/embedding/sentence_transformer_embedder.py

Sentence-transformers implementation of BaseEmbedder.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from app.embedding.base import BaseEmbedder


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(BaseEmbedder):

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> List[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    @property
    def dimension(self) -> int:
        return self._dimension
