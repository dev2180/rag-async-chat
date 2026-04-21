from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.embedding.sparse_embedder import SparseEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore


def test_doc_id_tracking():
    embedder = SentenceTransformerEmbedder()
    sparse_embedder = SparseEmbedder()
    store = QdrantVectorStore(collection_name="doc_test")

    store.create_collection(embedder.dimension, recreate=True)
    
    texts = ["Test document"]
    dense_vectors = embedder.embed_batch(texts)
    sparse_vectors = sparse_embedder.embed_batch(texts)
    
    payloads = [{
        "doc_id": "test_hash",
        "source_name": "test.pdf",
        "chunk_index": 0,
        "text": texts[0]
    }]

    store.upsert_vectors(dense_vectors, sparse_vectors, payloads)

    doc_ids = store.get_all_doc_ids()

    assert "test_hash" in doc_ids

    store.delete_by_doc_id("test_hash")

    doc_ids_after = store.get_all_doc_ids()

    assert "test_hash" not in doc_ids_after
