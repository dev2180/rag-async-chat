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
from app.utils.latency import LatencyTracker, QueryMetrics
from dataclasses import dataclass

@dataclass
class AnswerResult:
    answer: str
    metrics: QueryMetrics
    # Future additions: citations, eval

class RAGEngine:

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
    ):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str, session_id: str, top_k: int = 5) -> AnswerResult:

        metrics = QueryMetrics()
        memory = ChatMemory(session_id=session_id)
        history = memory.get_history()

        with LatencyTracker("Retrieval") as t:
            payloads = self.retriever.retrieve(query, top_k=top_k)
        metrics.retrieval_ms = t.duration_ms

        context_chunks = [p.get("text", "") for p in payloads]
        prompt = build_prompt(query, context_chunks, history)

        with LatencyTracker("LLM Generation") as t:
            response = self.llm.generate(prompt)
        metrics.llm_ms = t.duration_ms

        metrics.total_ms = metrics.retrieval_ms + metrics.llm_ms

        # Save conversation
        memory.add_message("user", query)
        memory.add_message("assistant", response)

        return AnswerResult(answer=response, metrics=metrics)
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