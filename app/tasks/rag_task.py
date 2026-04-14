"""
MODULE: app/tasks/rag_task.py

Responsibility:
    Defines worker-executed tasks.
    Contains internal payload schema.

Must NOT:
    - Contain API logic
    - Manage queues
"""

import logging
from pydantic import BaseModel
from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore
from app.rag.retriever import Retriever
from app.rag.engine import RAGEngine
from app.llm.ollama_client import OllamaClient
from app.config import OLLAMA_MODEL, QDRANT_COLLECTION

logger = logging.getLogger(__name__)

# Module-level singletons — persist across jobs within the same worker process
# Avoids reloading the ~90MB SentenceTransformer model on every request
_embedder = None
_vectorstore = None
_retriever = None
_llm = None
_engine = None


class RAGJobPayload(BaseModel):
    """
    Internal payload passed to worker.
    """
    session_id: str
    query: str


def _get_engine() -> RAGEngine:
    """Lazy-initialize and cache the RAG engine components."""
    global _embedder, _sparse_embedder, _vectorstore, _retriever, _llm, _engine

    if _engine is None:
        logger.info("Initializing RAG engine components (first job)...")
        from app.embedding.sparse_embedder import SparseEmbedder
        
        _embedder = SentenceTransformerEmbedder()
        _sparse_embedder = SparseEmbedder()
        _vectorstore = QdrantVectorStore(collection_name=QDRANT_COLLECTION)
        _retriever = Retriever(_embedder, _sparse_embedder, _vectorstore)
        _llm = OllamaClient(model=OLLAMA_MODEL)
        _engine = RAGEngine(_retriever, _llm)
        logger.info("RAG engine ready.")

    return _engine


def rag_chat_task(payload_dict: dict) -> str:
    """
    Executes full RAG pipeline inside worker.
    """
    payload = RAGJobPayload(**payload_dict)
    engine = _get_engine()

    logger.info(f"Processing query for session {payload.session_id}")

    result = engine.answer(
        query=payload.query,
        session_id=payload.session_id
    )
    return result.answer