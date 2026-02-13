"""
MODULE: app/tasks/rag_task.py

Responsibility:
    Defines worker-executed tasks.
    Contains internal payload schema.

Must NOT:
    - Contain API logic
    - Manage queues
"""

from pydantic import BaseModel
from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore
from app.rag.retriever import Retriever
from app.rag.engine import RAGEngine
from app.llm.ollama_client import OllamaClient


class RAGJobPayload(BaseModel):
    """
    Internal payload passed to worker.
    """
    session_id: str
    query: str


def rag_chat_task(payload_dict: dict) -> str:
    """
    Executes full RAG pipeline inside worker.
    """

    payload = RAGJobPayload(**payload_dict)

    embedder = SentenceTransformerEmbedder()
    vectorstore = QdrantVectorStore(collection_name="documents")
    retriever = Retriever(embedder, vectorstore)
    llm = OllamaClient(model="qwen2.5:7b-instruct")

    engine = RAGEngine(retriever, llm)

    return engine.answer(
        query=payload.query,
        session_id=payload.session_id
    )