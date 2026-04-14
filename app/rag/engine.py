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
from app.rag.evaluator import RetrievalMetrics, evaluate_retrieval
from app.rag.citations import Citation, build_citations
from app.rag.query_optimizer import QueryOptimizer
from typing import List
from dataclasses import dataclass

@dataclass
class AnswerResult:
    answer: str
    metrics: QueryMetrics
    eval: RetrievalMetrics
    citations: List[Citation]

class RAGEngine:

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
    ):
        self.retriever = retriever
        self.llm = llm
        self.optimizer = QueryOptimizer(llm)

    def answer(self, query: str, session_id: str, top_k: int = 5) -> AnswerResult:

        metrics = QueryMetrics()
        memory = ChatMemory(session_id=session_id)
        history = memory.get_history()

        optimized_query = self.optimizer.optimize(query, history)

        with LatencyTracker("Retrieval").measure() as t:
            payloads = self.retriever.retrieve(optimized_query, top_k=top_k)
        metrics.retrieval_ms = t.duration_ms
        eval_metrics = evaluate_retrieval(payloads)
        citations = build_citations(payloads)

        context_chunks = [p.get("text", "") for p in payloads]
        prompt = build_prompt(query, context_chunks, history)

        with LatencyTracker("LLM Generation").measure() as t:
            response = self.llm.generate(prompt)
        metrics.llm_ms = t.duration_ms

        metrics.total_ms = metrics.retrieval_ms + metrics.llm_ms

        # Save conversation
        memory.add_message("user", query)
        memory.add_message("assistant", response)

        return AnswerResult(answer=response, metrics=metrics, eval=eval_metrics, citations=citations)
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