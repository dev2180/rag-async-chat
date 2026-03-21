import pytest
from app.ingestion.chunker import chunk_text


def test_basic_chunking():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    # First chunk should be exactly chunk_size
    assert len(chunks[0]) == 400


def test_overlap_exists():
    """Verify overlapping content between consecutive chunks."""
    text = "abcdefghij" * 100  # 1000 chars, known pattern
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    # Last 80 chars of chunk[0] should equal first 80 chars of chunk[1]
    assert chunks[0][-80:] == chunks[1][:80]


def test_short_text():
    """Text shorter than chunk_size should produce one chunk."""
    text = "Hello world"
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


def test_empty_text():
    text = ""
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) == 0


def test_exact_chunk_size():
    """Exactly chunk_size chars produces 2 chunks due to overlap window."""
    text = "x" * 400
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    # step=320, positions: 0 (400 chars), 320 (80 chars)
    assert len(chunks) == 2
    assert chunks[0] == text


def test_chunk_size_plus_one():
    text = "x" * 401
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) == 2
    assert len(chunks[0]) == 400


def test_custom_params():
    text = "abcdefghij" * 10  # 100 chars
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    # Step = 30 - 10 = 20, so positions: 0, 20, 40, 60, 80
    assert len(chunks) == 5


def test_no_overlap():
    text = "x" * 100
    chunks = chunk_text(text, chunk_size=25, overlap=0)
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) == 25


def test_all_text_covered():
    """Every character in the original text appears in at least one chunk."""
    text = "abcdefghijklmnopqrstuvwxyz" * 5  # 130 chars
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    step = 40 - 10  # 30

    # Reconstruct all covered indices
    covered = set()
    for i, chunk in enumerate(chunks):
        start = i * step
        for j in range(len(chunk)):
            covered.add(start + j)

    # Every index from 0 to len(text)-1 should be covered
    for idx in range(len(text)):
        assert idx in covered, f"Index {idx} not covered by any chunk"


def test_single_char():
    text = "a"
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) == 1
    assert chunks[0] == "a"
