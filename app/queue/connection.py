"""
MODULE: app/queue/connection.py

Responsibility:
    Creates and exposes Valkey/Redis + RQ queue connections.
    Handles Windows compatibility (RQ workers use SimpleWorker on Windows).

Must NOT:
    - Contain business logic
    - Execute tasks
"""

import os
import logging
from redis import Redis
from app.config import REDIS_HOST, REDIS_PORT, REDIS_DB, QUEUE_NAME

logger = logging.getLogger(__name__)

# Redis (Valkey) connection
redis_conn = Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
)

# RQ Queue — works on all platforms
from rq import Queue

rag_queue = Queue(
    name=QUEUE_NAME,
    connection=redis_conn,
)

logger.info(f"Queue '{QUEUE_NAME}' initialized on {REDIS_HOST}:{REDIS_PORT}")