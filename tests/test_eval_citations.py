from app.rag.evaluator import evaluate_retrieval
from app.rag.citations import build_citations, format_citations_cli

# Mock payloads returned dynamically from retriever
MOCK_PAYLOADS = [
    {"score": 0.9, "source_name": "docA.pdf", "chunk_index": 1, "text": "This is doc A chunk 1 context which is slightly long."},
    {"score": 0.5, "source_name": "docA.pdf", "chunk_index": 5, "text": "Another chunk from exactly the same doc A. " * 5},
    {"score": 0.2, "source_name": "docB.pdf", "chunk_index": 2, "text": "Short chunk B."}
]

def test_evaluator_math():
    metrics = evaluate_retrieval(MOCK_PAYLOADS, threshold=0.3)
    
    # 0.9, 0.5, 0.2 -> avg = 1.6 / 3 = 0.533
    assert metrics.top_score == 0.9
    assert abs(metrics.avg_score - 0.533) < 0.01
    
    # 2 out of 3 above 0.3
    assert abs(metrics.coverage - 0.666) < 0.01
    assert metrics.num_results == 3
    assert set(metrics.source_docs) == {"docA.pdf", "docB.pdf"}

def test_empty_evaluator():
    metrics = evaluate_retrieval([])
    assert metrics.num_results == 0
    assert metrics.top_score == 0.0
    assert metrics.coverage == 0.0

def test_citations_builder():
    citations = build_citations(MOCK_PAYLOADS)
    assert len(citations) == 3
    
    assert citations[0].source == "docA.pdf"
    assert citations[0].relevance == 0.9
    assert citations[0].chunk_index == 1
    
    # Test formatting
    output = format_citations_cli(citations)
    assert "📎 Sources:" in output
    assert "[1] docA.pdf (chunk 1, relevance: 0.90)" in output
    assert "..." in output # second chunk should have an ellipsis
    assert "[3] docB.pdf (chunk 2, relevance: 0.20)" in output
