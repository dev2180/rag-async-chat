"""
MODULE: app/workers/worker.py

Responsibility:
    Worker bootstrap.
    Pulls jobs from queue and executes them.

Must NOT:
    - Accept HTTP requests
    - Contain RAG logic

NOTE: On Windows, use SimpleWorker:
    rq worker rag --worker-class rq.worker.SimpleWorker
"""

import os
import logging
from rq import Worker, SimpleWorker
from app.queue.connection import redis_conn, rag_queue
from app.config import setup_logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    setup_logging()

    # Use SimpleWorker on Windows (fork() not available)
    worker_class = SimpleWorker if os.name == "nt" else Worker

    logger.info(f"Starting worker (class={worker_class.__name__})...")

    worker = worker_class(
        queues=[rag_queue],
        connection=redis_conn,
    )
    worker.work()
