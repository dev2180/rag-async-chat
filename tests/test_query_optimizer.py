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
