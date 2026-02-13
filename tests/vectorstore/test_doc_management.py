from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore


def test_doc_id_tracking():
    embedder = SentenceTransformerEmbedder()
    store = QdrantVectorStore(collection_name="doc_test")

    store.create_collection(embedder.dimension)

    vectors = embedder.embed_batch(["Test document"])
    payloads = [{
        "doc_id": "test_hash",
        "source_name": "test.pdf",
        "chunk_index": 0,
        "text": "Test document"
    }]

    store.upsert_vectors(vectors, payloads)

    doc_ids = store.get_all_doc_ids()

    assert "test_hash" in doc_ids

    store.delete_by_doc_id("test_hash")

    doc_ids_after = store.get_all_doc_ids()

    assert "test_hash" not in doc_ids_after
