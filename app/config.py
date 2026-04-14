"""
MODULE: app/config.py

Centralized configuration for all services.
All hardcoded values consolidated here.
"""

import os
import logging

# --- Redis / Valkey ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# --- Qdrant ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")

# --- Ollama ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", 180))

# --- Embedding ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Queue ---
QUEUE_NAME = "rag"

# --- Paths ---
PDF_DIR = os.getenv("PDF_DIR", "data/pdfs")

# --- Chat ---
MAX_CHAT_MESSAGES = int(os.getenv("MAX_CHAT_MESSAGES", 10))

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "rag_app.log")

def setup_logging(console_level=None):
    """Configure logging for the application with file and console handlers."""
    import logging.handlers
    
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_FILE)

    root_logger = logging.getLogger()
    
    # Remove any existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Set base level
    root_logger.setLevel(logging.DEBUG) # Allows handlers to filter at their own level

    # 1. File Handler (Rotating)
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # 2. Console Handler
    target_console_level = console_level if console_level else LOG_LEVEL
    console_fmt = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, target_console_level.upper(), logging.INFO))
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)
