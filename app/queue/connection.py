"""
MODULE: app/queue/connection.py

Responsibility:
    Creates and exposes Valkey/Redis + RQ queue connections.

Must NOT:
    - Contain business logic
    - Execute tasks
"""

from redis import Redis
from rq import Queue

# Redis (Valkey) connection
redis_conn = Redis(
    host="localhost",
    port=6379,
    db=0,
)

# RQ queue
rag_queue = Queue(
    name="rag",
    connection=redis_conn,
)