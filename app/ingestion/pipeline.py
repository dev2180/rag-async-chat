"""
MODULE: app/ingestion/pipeline.py

Orchestrates PDF ingestion:
- Hash comparison
- Reconciliation
- Chunking
- Embedding
- Upserting to Qdrant
"""

from pathlib import Path
from app.ingestion.hasher import compute_file_hash
from app.ingestion.pdf_loader import load_pdf_text
from app.ingestion.chunker import chunk_text
from app.embedding.base import BaseEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore


class IngestionPipeline:

    def __init__(
        self,
        pdf_folder: Path,
        embedder: BaseEmbedder,
        vectorstore: QdrantVectorStore,
    ):
        self.pdf_folder = pdf_folder
        self.embedder = embedder
        self.vectorstore = vectorstore

    def run(self):

        print("\n[INFO] Starting ingestion pipeline...")

        pdf_files = list(self.pdf_folder.glob("*.pdf"))
        print(f"[INFO] Found {len(pdf_files)} PDF files")

        self.vectorstore.create_collection(self.embedder.dimension)

        existing_doc_ids = self.vectorstore.get_all_doc_ids()
        print(f"[INFO] Database contains {len(existing_doc_ids)} documents")

        current_hashes = set()

        for pdf_file in pdf_files:
            print(f"\n[INFO] Processing: {pdf_file.name}")

            file_hash = compute_file_hash(pdf_file)
            current_hashes.add(file_hash)

            if file_hash in existing_doc_ids:
                print("[INFO] No change detected. Skipping.")
                continue

            print("[INFO] New or updated file detected.")
            self._ingest_file(pdf_file, file_hash)

        for stored_doc_id in existing_doc_ids:
            if stored_doc_id not in current_hashes:
                print(f"[INFO] Removing deleted document: {stored_doc_id}")
                self.vectorstore.delete_by_doc_id(stored_doc_id)

        print("\n[INFO] Ingestion complete.")

    def _ingest_file(self, pdf_file: Path, file_hash: str):

        text = load_pdf_text(pdf_file)

        if not text.strip():
            print("[WARNING] No extractable text. Skipping.")
            return

        chunks = chunk_text(text)

        print(f"[INFO] Creating embeddings for {len(chunks)} chunks...")

        vectors = self.embedder.embed_batch(chunks)

        payloads = []

        for idx, chunk in enumerate(chunks):
            payloads.append({
                "doc_id": file_hash,
                "source_name": pdf_file.name,
                "chunk_index": idx,
                "text": chunk,
            })

        self.vectorstore.upsert_vectors(vectors, payloads)

        print("[INFO] Upsert complete.")
