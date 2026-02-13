"""
MODULE: app/workers/worker.py

Responsibility:
    Worker bootstrap.
    Pulls jobs from queue and executes them.

Must NOT:
    - Accept HTTP requests
    - Contain RAG logic
"""

from rq import Worker
from app.queue.connection import redis_conn, rag_queue


if __name__ == "__main__":
    worker = Worker(
        queues=[rag_queue],
        connection=redis_conn,
    )
    worker.work()
