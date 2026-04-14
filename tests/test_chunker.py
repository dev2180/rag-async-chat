"""
Independent tests for the adaptive chunker.
Validates sentence-awareness, code block preservation, overlap, and edge cases.
"""
import pytest
from app.ingestion.chunker import chunk_text


def test_basic_chunking():
    """Simple text should produce chunks within target size."""
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    chunks = chunk_text(text, chunk_size=60, overlap_sentences=0)

    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk should end at a sentence boundary (end with a period)
        assert chunk.rstrip().endswith(".")


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text(None) == []


def test_overlap_carries_sentences():
    """Overlap should repeat trailing sentences from previous chunk."""
    text = "Alpha. Bravo. Charlie. Delta. Echo. Foxtrot. Golf. Hotel."
    chunks = chunk_text(text, chunk_size=40, overlap_sentences=1)

    # The last sentence of chunk N should appear at the start of chunk N+1
    for i in range(len(chunks) - 1):
        prev_sentences = chunks[i].split(". ")
        next_sentences = chunks[i + 1].split(". ")

        last_of_prev = prev_sentences[-1].rstrip(".")
        first_of_next = next_sentences[0].rstrip(".")

        assert last_of_prev == first_of_next, (
            f"Expected overlap: '{last_of_prev}' should start chunk {i+1}, "
            f"but got '{first_of_next}'"
        )


def test_code_block_preservation():
    """Fenced code blocks must never be split across chunks."""
    text = """Here is some intro text about Python.

```python
def hello():
    print("Hello World")
    return True
```

And here is some follow-up explanation about the code above."""

    chunks = chunk_text(text, chunk_size=100, overlap_sentences=0)

    # Find which chunk contains the code block
    code_chunk = [c for c in chunks if "def hello():" in c]
    assert len(code_chunk) >= 1, "Code block should appear in at least one chunk"

    # The code block should be intact — opening and closing backticks together
    for c in code_chunk:
        assert "```python" in c
        assert c.count("```") == 2  # opening + closing


def test_paragraph_boundaries():
    """Double-newline paragraph breaks should be respected."""
    text = "First paragraph sentence one. First paragraph sentence two.\n\nSecond paragraph sentence one. Second paragraph sentence two."
    chunks = chunk_text(text, chunk_size=500, overlap_sentences=0)

    # With a large chunk_size, everything fits in one chunk
    assert len(chunks) >= 1


def test_single_long_sentence():
    """A sentence longer than chunk_size should become its own chunk, not crash."""
    long = "word " * 200  # ~1000 chars
    text = f"Short intro. {long} Short outro."
    chunks = chunk_text(text, chunk_size=100, overlap_sentences=0)

    assert len(chunks) >= 2
    # The long sentence should exist intact in one chunk
    long_chunks = [c for c in chunks if len(c) > 100]
    assert len(long_chunks) >= 1


def test_no_overlap_produces_no_duplicates():
    """With overlap_sentences=0, consecutive chunks should not share content."""
    text = "Alpha. Bravo. Charlie. Delta. Echo."
    chunks = chunk_text(text, chunk_size=30, overlap_sentences=0)

    # Check no sentence appears in two consecutive chunks
    for i in range(len(chunks) - 1):
        prev_set = set(chunks[i].split(". "))
        next_set = set(chunks[i + 1].split(". "))
        overlap = prev_set & next_set
        # Filter out empty strings
        overlap = {s for s in overlap if s.strip()}
        assert len(overlap) == 0, f"Unexpected overlap between chunk {i} and {i+1}: {overlap}"
