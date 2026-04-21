import pytest
import json
from app.rag.query_optimizer import QueryOptimizer

class MockLLM:
    def __init__(self, override_response: str = None):
        self.override_response = override_response
        self.last_prompt = ""
        
    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.override_response or '["optimized test query"]'

def test_query_optimizer_empty_history():
    mock_llm = MockLLM(override_response='["what is it?", "define it"]')
    optimizer = QueryOptimizer(mock_llm)
    
    result = optimizer.optimize("what is it?", [])
    assert result == ["what is it?", "define it"]
    assert "JSON OUTPUT:" in mock_llm.last_prompt

def test_query_optimizer_with_history():
    mock_llm = MockLLM(override_response='["what is node.js?", "node.js definition"]')
    optimizer = QueryOptimizer(mock_llm)
    
    history = [
        {"role": "user", "content": "Tell me about Node"},
        {"role": "assistant", "content": "Node.js is a runtime."}
    ]
    
    result = optimizer.optimize("what is it?", history)
    
    assert result == ["what is node.js?", "node.js definition"]
    assert "Tell me about Node" in mock_llm.last_prompt
    assert "USER QUERY: what is it?" in mock_llm.last_prompt

def test_query_optimizer_strips_markdown_json():
    mock_llm = MockLLM(override_response='```json\n["query1", "query2"]\n```')
    optimizer = QueryOptimizer(mock_llm)
    
    result = optimizer.optimize("query", [{"role": "user", "content": "hi"}])
    assert result == ["query1", "query2"]

def test_query_optimizer_fallback_on_bad_json():
    long_response = "This is not json " * 5
    mock_llm = MockLLM(override_response=long_response)
    optimizer = QueryOptimizer(mock_llm)
    
    result = optimizer.optimize("short query", [{"role": "user", "content": "hi"}])
    assert result == ["short query"]

def test_query_optimizer_limits_to_three():
    mock_llm = MockLLM(override_response='["one", "two", "three", "four"]')
    optimizer = QueryOptimizer(mock_llm)
    
    result = optimizer.optimize("query", [{"role": "user", "content": "hi"}])
    assert len(result) == 3
    assert result == ["one", "two", "three"]


def test_query_optimizer_handles_multiple_arrays():
    """Stage 2: LLM returns multiple separate JSON arrays instead of one flat array.
    This is the real-world failure seen with llama3.2 producing one array per variant."""
    response = (
        '["Lencho character", "The Old Man and the Sea"]\n'
        '["What is Lencho like?", "Why is Lencho important?"]\n'
        '["Lencho biography"]'
    )
    mock_llm = MockLLM(override_response=response)
    optimizer = QueryOptimizer(mock_llm)

    result = optimizer.optimize("who was lencho", [])

    # Should extract and merge strings from all arrays, capping at 3
    assert len(result) == 3
    assert "Lencho character" in result
    assert "What is Lencho like?" in result


def test_query_optimizer_deduplicates_across_arrays():
    """Stage 2: Duplicate strings appearing in multiple arrays are included only once."""
    response = '["node.js runtime"]\n["node.js runtime", "what is node"]'
    mock_llm = MockLLM(override_response=response)
    optimizer = QueryOptimizer(mock_llm)

    result = optimizer.optimize("what is node", [])

    assert result.count("node.js runtime") == 1  # no duplicates
