"""
MODULE: app/rag/engine.py

Responsibility:
    Orchestrates the RAG pipeline:
    retrieve → prompt → generate → return answer.

Must NOT:
    - Talk to Redis
    - Know about FastAPI
"""

from app.rag.retriever import Retriever
from app.rag.prompt import build_prompt
from app.llm.base import BaseLLM
from app.chat.memory import ChatMemory


class RAGEngine:

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
    ):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str, session_id: str, top_k: int = 5) -> str:

        memory = ChatMemory(session_id=session_id)

        history = memory.get_history()

        payloads = self.retriever.retrieve(query, top_k=top_k)

        context_chunks = [p.get("text", "") for p in payloads]

        prompt = build_prompt(query, context_chunks, history)

        response = self.llm.generate(prompt)

        # Save conversation
        memory.add_message("user", query)
        memory.add_message("assistant", response)

        return response
    def stream_answer(self, query: str, session_id: str, top_k: int = 5):

        memory = ChatMemory(session_id=session_id)
        history = memory.get_history()

        payloads = self.retriever.retrieve(query, top_k=top_k)
        context_chunks = [p.get("text", "") for p in payloads]

        prompt = build_prompt(query, context_chunks, history)

        full_response = ""

        try:
            for token in self.llm.stream_generate(prompt):
                full_response += token
                yield token
        finally:
            if full_response:
                memory.add_message("user", query)
                memory.add_message("assistant", full_response)