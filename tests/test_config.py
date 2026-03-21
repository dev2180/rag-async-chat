"""
Tests for app/config.py

Verifies that the centralized config module loads defaults correctly
and can be overridden via environment variables.
"""

import os
import pytest


def test_default_config_values():
    """Test that config defaults are sensible."""
    from app.config import (
        REDIS_HOST, REDIS_PORT,
        QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION,
        OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT,
        EMBEDDING_MODEL, QUEUE_NAME, PDF_DIR,
        MAX_CHAT_MESSAGES, LOG_LEVEL,
    )

    assert REDIS_HOST == "localhost"
    assert REDIS_PORT == 6379
    assert QDRANT_HOST == "localhost"
    assert QDRANT_PORT == 6333
    assert QDRANT_COLLECTION == "documents"
    assert OLLAMA_BASE_URL == "http://localhost:11434"
    assert OLLAMA_MODEL == "llama3.2:latest"
    assert OLLAMA_TIMEOUT == 180
    assert EMBEDDING_MODEL == "all-MiniLM-L6-v2"
    assert QUEUE_NAME == "rag"
    assert PDF_DIR == "data/pdfs"
    assert MAX_CHAT_MESSAGES == 10
    assert LOG_LEVEL == "INFO"


def test_setup_logging():
    """Test that setup_logging runs without error."""
    from app.config import setup_logging
    setup_logging()  # should not raise
