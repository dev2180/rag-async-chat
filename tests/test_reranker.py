import pytest
from app.rag.reranker import CrossEncoderReranker

@pytest.fixture(scope="module")
def reranker():
    # Use a tiny mock or tiny model for testing? 
    # Actually, initializing the real MS-Marco model takes seconds and ensures actual integration shape.
    # To keep tests ultra-fast, we could mock the cross-encoder, but using the real 'cross-encoder/ms-marco-MiniLM-L-6-v2' 
    # is only ~80MB and assures the tensor shapes align.
    return CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_reranker_sorting(reranker):
    query = "How to run the python server?"
    
    payloads = [
        {"id": 1, "text": "To bake bread, you need flour and yeast."}, # Irrelevant
        {"id": 2, "text": "Python is a snake found in tropical areas."}, # Partly relevant keyword, wrong semantic
        {"id": 3, "text": "Start the backend by running python server.py in the terminal."}, # Highly relevant
    ]
    
    # Intentionally misordered by initial vector search (hypothetical)
    reranked = reranker.rerank(query, payloads, top_k=3)
    
    # Result should be exactly 3 payloads
    assert len(reranked) == 3
    
    # ID 3 should be ranked first because it is highly semantically relevant
    assert reranked[0]["id"] == 3
    
    # ID 1 should be ranked last because it is completely irrelevant
    assert reranked[-1]["id"] == 1
    
    # Scores should be descending
    assert reranked[0]["score"] > reranked[1]["score"]
    assert reranked[1]["score"] > reranked[2]["score"]


def test_reranker_truncation(reranker):
    query = "test"
    payloads = [{"text": f"payload {i}"} for i in range(10)]
    
    reranked = reranker.rerank(query, payloads, top_k=2)
    assert len(reranked) == 2


def test_reranker_empty_input(reranker):
    assert reranker.rerank("test", []) == []
