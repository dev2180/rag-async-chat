"""
Independent tests for SparseEmbedder.
Validates that fastembed produces valid sparse vector outputs (indices + values).
"""
import pytest
from app.embedding.sparse_embedder import SparseEmbedder


@pytest.fixture(scope="module")
def embedder():
    """Module-scoped so the BM25 model loads only once across all tests."""
    return SparseEmbedder()


def test_embed_single_text(embedder):
    indices, values = embedder.embed_text("Node.js is a JavaScript runtime")

    # Must return native Python lists, not numpy arrays
    assert isinstance(indices, list), f"Expected list, got {type(indices)}"
    assert isinstance(values, list), f"Expected list, got {type(values)}"

    # Non-empty for a real sentence
    assert len(indices) > 0
    assert len(values) > 0

    # Lengths must match (one weight per token index)
    assert len(indices) == len(values)

    # Indices should be ints, values should be floats
    assert all(isinstance(i, int) for i in indices)
    assert all(isinstance(v, float) for v in values)

    # All values should be positive (BM25 weights)
    assert all(v > 0 for v in values)


def test_embed_batch(embedder):
    texts = [
        "Machine learning is a subset of AI",
        "Docker containers simplify deployment",
    ]
    results = embedder.embed_batch(texts)

    assert len(results) == 2

    for indices, values in results:
        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values)
        assert len(indices) > 0


def test_different_texts_produce_different_vectors(embedder):
    """Distinct inputs should yield different sparse representations."""
    idx_a, _ = embedder.embed_text("Python programming language")
    idx_b, _ = embedder.embed_text("Cooking pasta in olive oil")

    # The index sets should differ meaningfully
    assert set(idx_a) != set(idx_b)
