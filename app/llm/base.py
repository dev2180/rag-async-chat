"""
MODULE: app/llm/base.py

Defines the LLM interface.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def stream_generate(self, prompt: str):
        pass
