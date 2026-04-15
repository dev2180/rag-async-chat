"""
Independent tests for the context compressor.
Validates score filtering, keyword overlap, safety fallback, and edge cases.
"""
import pytest
from app.rag.compressor import compress_context, _extract_keywords


MOCK_PAYLOADS = [
    {"score": 0.9, "text": "Node.js is a JavaScript runtime environment built on V8 engine."},
    {"score": 0.5, "text": "Docker containers simplify application deployment workflows."},
    {"score": 0.1, "text": "The weather today is sunny with clear skies."},
    {"score": 0.3, "text": "JavaScript runs on the server side using Node runtime."},
]


def test_score_threshold_filtering():
    """Chunks below score_threshold should be removed."""
    result = compress_context(MOCK_PAYLOADS, "node.js", score_threshold=0.25)
    scores = [p["score"] for p in result]
    assert all(s >= 0.25 for s in scores)
    assert 0.1 not in scores


def test_keyword_overlap_filtering():
    """Chunks with zero keyword overlap should be removed."""
    result = compress_context(MOCK_PAYLOADS, "node.js runtime", score_threshold=0.0, min_keyword_overlap=1)
    texts = [p["text"] for p in result]
    
    # Docker chunk has no keyword overlap with "node.js runtime"
    assert not any("Docker" in t for t in texts)
    
    # Node chunks should remain
    assert any("Node" in t for t in texts)


def test_safety_fallback_keeps_best():
    """If all chunks get filtered, keep the single best one."""
    payloads = [
        {"score": 0.05, "text": "Completely unrelated content about cooking."},
        {"score": 0.08, "text": "Another irrelevant chunk about gardening."},
    ]
    result = compress_context(payloads, "quantum physics", score_threshold=0.5)
    
    assert len(result) == 1
    assert result[0]["score"] == 0.08  # Best of the bunch


def test_empty_input():
    assert compress_context([], "anything") == []


def test_extract_keywords():
    keywords = _extract_keywords("What is Node.js and how does it work?")
    assert "node" in keywords
    assert "work" in keywords
    # Stopwords should be excluded
    assert "what" not in keywords
    assert "is" not in keywords
    assert "and" not in keywords


def test_compression_preserves_order():
    """Passing chunks should maintain their original order."""
    payloads = [
        {"score": 0.9, "text": "First chunk about Python programming."},
        {"score": 0.7, "text": "Second chunk about Python libraries."},
        {"score": 0.5, "text": "Third chunk about Python frameworks."},
    ]
    result = compress_context(payloads, "Python programming", score_threshold=0.0)
    
    assert result[0]["text"].startswith("First")
    assert result[-1]["text"].startswith("Third")
