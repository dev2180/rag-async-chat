import os
import sys
import uuid
import logging
from pathlib import Path

# --- PRE-FLIGHT CHECKS ---
# 1. Virtual Environment Check
if sys.prefix == sys.base_prefix:
    print("\n\033[93m⚠️  Dude, you need to run this in your virtual environment!\033[0m")
    print("Run this command first: \033[96mvenv\\Scripts\\activate\033[0m (on Windows) or \033[96msource venv/bin/activate\033[0m (on Mac/Linux)\n")
    sys.exit(1)

# Now it's safe to import third party libraries (because we are in the venv)
try:
    import requests
    from redis import Redis
    from app.config import (
        setup_logging, PDF_DIR, 
        REDIS_HOST, REDIS_PORT, 
        QDRANT_HOST, QDRANT_PORT, 
        OLLAMA_BASE_URL
    )
    from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
    from app.vectorstore.qdrant_client import QdrantVectorStore
    from app.rag.retriever import Retriever
    from app.llm.ollama_client import OllamaClient
    from app.rag.engine import RAGEngine
    from app.ingestion.pipeline import IngestionPipeline
except ImportError as e:
    print(f"\n\033[91m⚠️  Missing dependency: {e}\033[0m")
    print("Dude, did you forget to install the requirements?")
    print("Run: \033[96mpip install -r requirements.txt\033[0m\n")
    sys.exit(1)

logger = logging.getLogger(__name__)


def check_services():
    """Verify all required Docker services are actually running before crashing later."""
    print("\n\033[94m🔍 Checking required services...\033[0m")
    
    # Check Valkey (Redis)
    try:
        r = Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=2)
        r.ping()
        print(" ✅ \033[92mValkey (Redis)\033[0m is running")
    except Exception:
        print("\n\033[91m❌ Dude, Valkey (Redis) is not running!\033[0m")
        print("Run it first with: \033[96mdocker run -d --name valkey -p 6379:6379 valkey/valkey\033[0m\n")
        sys.exit(1)
        
    # Check Qdrant
    try:
        requests.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}", timeout=2)
        print(" ✅ \033[92mQdrant Vector DB\033[0m is running")
    except requests.exceptions.RequestException:
        print("\n\033[91m❌ Dude, Qdrant Vector DB is not running!\033[0m")
        print("Run it first with: \033[96mdocker run -d --name rag-vector-db -p 6333:6333 qdrant/qdrant\033[0m\n")
        sys.exit(1)
        
    # Check Ollama
    try:
        requests.get(f"{OLLAMA_BASE_URL}", timeout=2)
        print(" ✅ \033[92mOllama Client\033[0m is running")
    except requests.exceptions.RequestException:
        print("\n\033[91m❌ Dude, Ollama is not running!\033[0m")
        print("Start Ollama on your machine or run the docker container first.\n")
        sys.exit(1)
    
    print("-" * 40)


def check_pdfs():
    if not os.path.exists(PDF_DIR):
        print(f"\n\033[93m⚠️  Folder '{PDF_DIR}' does not exist.\033[0m")
        sys.exit(1)

    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdfs:
        print("\n\033[93m⚠️  No PDFs found.\033[0m")
        print(f"Dude, please upload your PDFs into '{PDF_DIR}' and run again.")
        sys.exit(1)

    print(f"\n\033[94m📄 Found {len(pdfs)} PDFs:\033[0m")
    for pdf in pdfs:
        print(f" - {pdf}")


def run_ingestion(embedder, vectorstore):
    print("\n\033[95m🚀 Running ingestion pipeline...\033[0m")

    pdf_folder = Path(PDF_DIR)
    pipeline = IngestionPipeline(
        pdf_folder=pdf_folder,
        embedder=embedder,
        vectorstore=vectorstore,
    )
    pipeline.run()

    print("\033[92m✅ Ingestion completed.\033[0m\n")


def start_chat(embedder, vectorstore):
    print("\n" + "="*40)
    print(" \033[96mStarting RAG CLI Chat Session...\033[0m")
    print(" Type '\033[91mexit\033[0m' or '\033[91mquit\033[0m' to end the session.")
    print("="*40 + "\n")

    session_id = str(uuid.uuid4())
    retriever = Retriever(embedder=embedder, vectorstore=vectorstore)
    llm = OllamaClient()
    engine = RAGEngine(retriever=retriever, llm=llm)

    while True:
        try:
            question = input("\033[93mYou > \033[0m")
        except EOFError:
            break

        if not question.strip():
            continue

        if question.lower().strip() in ["exit", "quit"]:
            print("\n\033[92m👋 Exiting session. See ya!\033[0m\n")
            break

        try:
            result = engine.answer(question, session_id=session_id)
            print(f"\n\033[96mAssistant >\033[0m {result.answer}\n")
            print(f"⏱️  Retrieval: {result.metrics.retrieval_ms:.0f}ms | LLM: {result.metrics.llm_ms:.1f}ms | Total: {result.metrics.total_ms:.1f}ms")
            print("-" * 40)
        except ConnectionError as e:
            print(f"\n\033[91m❌ Connection error: {e}\033[0m\n")
        except RuntimeError as e:
            print(f"\n\033[91m❌ Error: {e}\033[0m\n")
        except Exception as e:
            print(f"\n\033[91m❌ Unexpected error: {e}\033[0m\n")
            logger.exception("Unexpected error during chat")


def main():
    setup_logging(console_level="WARNING")

    print("\n\033[1m\033[95m==== LOCAL ASYNC RAG CLI ====\033[0m")

    check_services()
    check_pdfs()

    print("\n\033[94m⏳ Loading AI Models (this takes a few seconds)...\033[0m")
    
    # Create shared instances
    embedder = SentenceTransformerEmbedder()
    vectorstore = QdrantVectorStore()

    run_ingestion(embedder, vectorstore)
    start_chat(embedder, vectorstore)


if __name__ == "__main__":
    main()