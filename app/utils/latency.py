"""
MODULE: app/utils/latency.py
Provides latency tracking context managers for the RAG pipeline.
"""
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    query_rewrite_ms: float = 0.0
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    compress_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

class LatencyTracker:
    def __init__(self, name: str):
        self.name = name
        self.duration_ms = 0.0

    @contextmanager
    def measure(self):
        start = time.perf_counter()
        try:
            yield self
        finally:
            end = time.perf_counter()
            self.duration_ms = (end - start) * 1000
            logger.debug(f"{self.name} took {self.duration_ms:.2f} ms")
