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
from app.rag.compressor import compress_context
from app.rag.reranker import CrossEncoderReranker
from app.utils.trace import TraceLogger
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
        reranker: CrossEncoderReranker = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.optimizer = QueryOptimizer(llm)
        self.reranker = reranker

    def answer(self, query: str, session_id: str, top_k: int = 5) -> AnswerResult:
        tracer = TraceLogger(query)

        metrics = QueryMetrics()
        memory = ChatMemory(session_id=session_id)
        history = memory.get_history()

        with LatencyTracker("Query Optimization").measure() as t:
            optimized_queries = self.optimizer.optimize(query, history)
        metrics.query_rewrite_ms = t.duration_ms
        tracer.add_step("Query Expansion", f"Generated variants:\n- " + "\n- ".join(optimized_queries))

        payloads = []
        seen_texts = set()
        retrieval_tracker = LatencyTracker("Retrieval")
        with retrieval_tracker.measure():
            for q in optimized_queries:
                # Retrieve for each query variation
                results = self.retriever.retrieve(q, top_k=top_k)
                for r in results:
                    text = r.get("text", "")
                    if text not in seen_texts:
                        seen_texts.add(text)
                        payloads.append(r)
        metrics.retrieval_ms = retrieval_tracker.duration_ms

        # Sort payloads by score descending
        payloads.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        # We might have up to top_k * 3 chunks now. Let's keep the best ones before eval.
        payloads = payloads[:top_k * 2]
        tracer.add_step("Multi-Query Retrieval", f"Pooled {len(payloads)} unique candidate chunks from vector store.")

        if self.reranker:
            with LatencyTracker("Reranking").measure() as rerank_tracker:
                payloads = self.reranker.rerank(query, payloads, top_k=top_k)
            metrics.rerank_ms = rerank_tracker.duration_ms

            rerank_details = "\n".join([f"Score: {p.get('score', 0):.2f} | Chunk {p.get('id', '?')} -> {p.get('text', '')[:100]}..." for p in payloads])
            tracer.add_step("CrossEncoder Reranking", rerank_details)
        else:
            payloads = payloads[:top_k]
        eval_metrics = evaluate_retrieval(payloads)
        citations = build_citations(payloads)

        with LatencyTracker("Compression").measure() as t:
            # Combine queries for keyword extraction
            combined_queries = " ".join(optimized_queries)
            compressed = compress_context(payloads, combined_queries)
        metrics.compress_ms = t.duration_ms
        
        dropped = len(payloads) - len(compressed)
        tracer.add_step("Context Compression", f"Dropped {dropped} chunks due to low keyword/score overlap. Remaining: {len(compressed)}")

        context_chunks = [p.get("text", "") for p in compressed]
        prompt = build_prompt(query, context_chunks, history)

        with LatencyTracker("LLM Generation").measure() as t:
            response = self.llm.generate(prompt)
        metrics.llm_ms = t.duration_ms

        metrics.total_ms = metrics.query_rewrite_ms + metrics.retrieval_ms + metrics.rerank_ms + metrics.compress_ms + metrics.llm_ms

        # Save conversation
        memory.add_message("user", query)
        memory.add_message("assistant", response)

        tracer.add_step("LLM Generation", f"Final Response:\n{response}")
        tracer.save()

        return AnswerResult(answer=response, metrics=metrics, eval=eval_metrics, citations=citations)

    def stream_answer(self, query: str, session_id: str, top_k: int = 5):
        """
        Streaming variant of the RAG pipeline.
        Uses multi-query expansion and hybrid retrieval for consistency with answer().
        Yields response tokens as they are generated.
        """
        memory = ChatMemory(session_id=session_id)
        history = memory.get_history()

        # Multi-query expansion (same as non-streaming path)
        optimized_queries = self.optimizer.optimize(query, history)

        # Hybrid retrieval across all query variants, deduplicated
        payloads: list = []
        seen_texts: set = set()
        for q in optimized_queries:
            results = self.retriever.retrieve(q, top_k=top_k)
            for r in results:
                text = r.get("text", "")
                if text not in seen_texts:
                    seen_texts.add(text)
                    payloads.append(r)

        payloads.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        payloads = payloads[:top_k * 2]

        # Optional reranking
        if self.reranker:
            payloads = self.reranker.rerank(query, payloads, top_k=top_k)
        else:
            payloads = payloads[:top_k]

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