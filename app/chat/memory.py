"""
MODULE: app/chat/memory.py

Handles chat history storage in Redis.
"""

import json
import logging
from typing import List, Dict
from app.queue.connection import redis_conn
from app.config import MAX_CHAT_MESSAGES

logger = logging.getLogger(__name__)


class ChatMemory:

    def __init__(self, session_id: str, max_messages: int = MAX_CHAT_MESSAGES):
        self.session_id = session_id
        self.key = f"chat:{session_id}"
        self.max_messages = max_messages

    def add_message(self, role: str, content: str):
        """
        role: 'user' or 'assistant'
        """
        message = {"role": role, "content": content}

        redis_conn.rpush(self.key, json.dumps(message))
        redis_conn.ltrim(self.key, -self.max_messages, -1)

    def get_history(self) -> List[Dict]:
        messages = redis_conn.lrange(self.key, 0, -1)
        return [json.loads(m) for m in messages]

    def clear(self):
        redis_conn.delete(self.key)
        logger.info(f"Cleared chat history for session {self.session_id}")
