"""
Tests for the adaptive chunker (ingestion-level integration tests).
These tests use the CURRENT API: chunk_text(text, chunk_size, overlap_sentences).

NOTE: The old naive `overlap=N` character-slicer parameter no longer exists.
The chunker now uses NLTK sentence-boundary aware splitting with sentence-level overlap.
"""
import pytest
from app.ingestion.chunker import chunk_text


def test_basic_chunking_returns_multiple_chunks():
    """Long text should be split into multiple chunks."""
    # Create text with many sentences to force chunking
    text = "This is sentence number {i}. " * 30
    text = "".join(f"This is sentence number {i}. " for i in range(30))
    chunks = chunk_text(text, chunk_size=100, overlap_sentences=0)
    assert len(chunks) > 1


def test_short_text_produces_one_chunk():
    """Text shorter than chunk_size should produce exactly one chunk."""
    text = "Hello world. This is a short sentence."
    chunks = chunk_text(text, chunk_size=500, overlap_sentences=0)
    assert len(chunks) == 1
    assert "Hello world" in chunks[0]


def test_empty_text():
    """Empty or whitespace-only text should produce no chunks."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text(None) == []


def test_overlap_carries_sentences():
    """With overlap_sentences=1, the last sentence of chunk N appears at start of chunk N+1."""
    text = "Alpha. Bravo. Charlie. Delta. Echo. Foxtrot. Golf. Hotel."
    chunks = chunk_text(text, chunk_size=40, overlap_sentences=1)

    if len(chunks) > 1:
        prev_sentences = chunks[0].split(". ")
        next_sentences = chunks[1].split(". ")
        last_of_prev = prev_sentences[-1].rstrip(".")
        first_of_next = next_sentences[0].rstrip(".")
        assert last_of_prev == first_of_next


def test_no_overlap_no_duplicates():
    """With overlap_sentences=0, consecutive chunks share no sentences."""
    text = "Alpha. Bravo. Charlie. Delta. Echo."
    chunks = chunk_text(text, chunk_size=30, overlap_sentences=0)

    for i in range(len(chunks) - 1):
        prev_set = set(s.strip().rstrip(".") for s in chunks[i].split(". ") if s.strip())
        next_set = set(s.strip().rstrip(".") for s in chunks[i + 1].split(". ") if s.strip())
        overlap = prev_set & next_set - {""}
        assert len(overlap) == 0, f"Unexpected duplicate sentences between chunk {i} and {i+1}: {overlap}"


def test_code_block_preserved():
    """Fenced code blocks must not be split across chunks."""
    text = """Intro text about the API.

```python
def hello():
    print("Hello World")
    return True
```

Follow-up explanation here."""

    chunks = chunk_text(text, chunk_size=100, overlap_sentences=0)
    code_chunks = [c for c in chunks if "def hello():" in c]
    assert len(code_chunks) >= 1
    for c in code_chunks:
        assert "```python" in c
        assert c.count("```") == 2


def test_single_long_sentence_becomes_own_chunk():
    """A single sentence longer than chunk_size should not crash."""
    long_sentence = "word " * 200  # ~1000 chars
    chunks = chunk_text(long_sentence, chunk_size=100, overlap_sentences=0)
    assert len(chunks) >= 1
    long_chunks = [c for c in chunks if len(c) > 100]
    assert len(long_chunks) >= 1
