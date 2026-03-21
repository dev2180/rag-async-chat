import sys

print("Testing app.config...")
from app.config import setup_logging, PDF_DIR
print("Testing app.embedding...")
from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
print("Testing app.vectorstore...")
from app.vectorstore.qdrant_client import QdrantVectorStore
print("Testing app.rag.retriever...")
from app.rag.retriever import Retriever
print("Testing app.llm.ollama_client...")
from app.llm.ollama_client import OllamaClient
print("Testing app.rag.engine...")
from app.rag.engine import RAGEngine
print("Testing app.ingestion.pipeline...")
from app.ingestion.pipeline import IngestionPipeline

print("ALL CLEAR")
