from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.embedding.sparse_embedder import SparseEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore


def test_qdrant_insert_and_search():
    embedder = SentenceTransformerEmbedder()
    sparse_embedder = SparseEmbedder()
    store = QdrantVectorStore(collection_name="test_collection")

    store.create_collection(embedder.dimension, recreate=True)

    texts = ["The sky is blue", "The sun is bright"]
    dense_vectors = embedder.embed_batch(texts)
    sparse_vectors = sparse_embedder.embed_batch(texts)

    payloads = [{"text": t} for t in texts]

    store.upsert_vectors(dense_vectors, sparse_vectors, payloads)

    query = "blue sky"
    q_dense = embedder.embed_text(query)
    q_sparse = sparse_embedder.embed_text(query)

    results = store.search(q_dense, q_sparse, top_k=2)

    assert len(results) > 0
