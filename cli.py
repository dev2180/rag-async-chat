import os
import sys
import uuid
import logging
from pathlib import Path
from app.config import setup_logging, PDF_DIR
from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore
from app.rag.retriever import Retriever
from app.llm.ollama_client import OllamaClient
from app.rag.engine import RAGEngine
from app.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


def check_pdfs():
    if not os.path.exists(PDF_DIR):
        print(f"\n Folder '{PDF_DIR}' does not exist.")
        sys.exit(1)

    pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdfs:
        print("\n No PDFs found.")
        print(f" Please upload your PDFs into '{PDF_DIR}' and run again.")
        sys.exit(1)

    print("\n📄 Found PDFs:")
    for pdf in pdfs:
        print(f" - {pdf}")

    return pdfs


def run_ingestion(embedder, vectorstore):
    """Run ingestion with shared embedder and vectorstore."""
    print("\n🚀 Running ingestion pipeline...")

    pdf_folder = Path(PDF_DIR)

    pipeline = IngestionPipeline(
        pdf_folder=pdf_folder,
        embedder=embedder,
        vectorstore=vectorstore,
    )

    pipeline.run()

    print("✅ Ingestion completed.\n")


def start_chat(embedder, vectorstore):
    """Start chat with shared embedder and vectorstore."""
    print(" Starting RAG CLI session...")
    print("Type 'exit' to quit.\n")

    session_id = str(uuid.uuid4())

    retriever = Retriever(embedder=embedder, vectorstore=vectorstore)
    llm = OllamaClient()

    engine = RAGEngine(
        retriever=retriever,
        llm=llm,
    )

    while True:
        question = input("You > ")

        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting session.")
            break

        try:
            answer = engine.answer(question, session_id=session_id)
            print(f"\nAssistant > {answer}\n")
        except ConnectionError as e:
            print(f"\n❌ Connection error: {e}\n")
        except RuntimeError as e:
            print(f"\n❌ Error: {e}\n")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")
            logger.exception("Unexpected error during chat")


def main():
    setup_logging()
    print("==== LOCAL ASYNC RAG CLI ====")

    check_pdfs()

    # Create shared instances — loaded once, used by both ingestion and chat
    embedder = SentenceTransformerEmbedder()
    vectorstore = QdrantVectorStore()

    run_ingestion(embedder, vectorstore)
    start_chat(embedder, vectorstore)


if __name__ == "__main__":
    main()