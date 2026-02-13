from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder
from app.vectorstore.qdrant_client import QdrantVectorStore


def test_qdrant_insert_and_search():
    embedder = SentenceTransformerEmbedder()
    store = QdrantVectorStore(collection_name="test_collection")

    store.create_collection(embedder.dimension)

    texts = ["The sky is blue", "The sun is bright"]
    vectors = embedder.embed_batch(texts)

    payloads = [{"text": t} for t in texts]

    store.upsert_vectors(vectors, payloads)

    query_vector = embedder.embed_text("blue sky")

    results = store.search(query_vector, top_k=2)

    assert len(results) > 0
