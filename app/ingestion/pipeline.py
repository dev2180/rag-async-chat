"""
MODULE: app/ingestion/pipeline.py

Orchestrates PDF ingestion:
- Hash comparison
- Reconciliation
- Chunking
- Embedding
- Upserting to Qdrant
"""

import logging
from pathlib import Path
from app.ingestion.hasher import compute_file_hash
from app.ingestion.pdf_loader import load_pdf_text
from app.ingestion.chunker import chunk_text
from app.embedding.base import BaseEmbedder
from app.embedding.sparse_embedder import SparseEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore

logger = logging.getLogger(__name__)


class IngestionPipeline:

    def __init__(
        self,
        pdf_folder: Path,
        embedder: BaseEmbedder,
        sparse_embedder: SparseEmbedder,
        vectorstore: QdrantVectorStore,
    ):
        self.pdf_folder = pdf_folder
        self.embedder = embedder
        self.sparse_embedder = sparse_embedder
        self.vectorstore = vectorstore

    def run(self):

        logger.info("Starting ingestion pipeline...")

        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        self.vectorstore.create_collection(self.embedder.dimension)

        existing_doc_ids = self.vectorstore.get_all_doc_ids()
        logger.info(f"Database contains {len(existing_doc_ids)} documents")

        current_hashes = set()

        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")

            file_hash = compute_file_hash(pdf_file)
            current_hashes.add(file_hash)

            if file_hash in existing_doc_ids:
                logger.info("No change detected. Skipping.")
                continue

            logger.info("New or updated file detected.")
            self._ingest_file(pdf_file, file_hash)

        for stored_doc_id in existing_doc_ids:
            if stored_doc_id not in current_hashes:
                logger.info(f"Removing deleted document: {stored_doc_id}")
                self.vectorstore.delete_by_doc_id(stored_doc_id)

        logger.info("Ingestion complete.")

    def _ingest_file(self, pdf_file: Path, file_hash: str):

        text = load_pdf_text(pdf_file)

        if not text.strip():
            logger.warning(f"No extractable text in {pdf_file.name}. Skipping.")
            return

        chunks = chunk_text(text)

        logger.info(f"Creating embeddings for {len(chunks)} chunks...")

        dense_vectors = self.embedder.embed_batch(chunks)
        sparse_vectors = self.sparse_embedder.embed_batch(chunks)

        payloads = []

        for idx, chunk in enumerate(chunks):
            payloads.append({
                "doc_id": file_hash,
                "source_name": pdf_file.name,
                "chunk_index": idx,
                "text": chunk,
            })

        self.vectorstore.upsert_vectors(dense_vectors, sparse_vectors, payloads)

        logger.info("Upsert complete.")
