"""
MODULE: app/embedding/base.py

Defines the embedding interface.

Must NOT:
    - Depend on specific libraries
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass
