from app.embedding.sentence_transformer_embedder import SentenceTransformerEmbedder


def test_embedding_dimension():
    embedder = SentenceTransformerEmbedder()
    assert embedder.dimension > 0


def test_embed_text():
    embedder = SentenceTransformerEmbedder()
    vector = embedder.embed_text("Hello world")
    assert len(vector) == embedder.dimension
