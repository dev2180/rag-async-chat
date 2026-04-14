import pytest
from app.rag.query_optimizer import QueryOptimizer

class MockLLM:
    def __init__(self, override_response: str = None):
        self.override_response = override_response
        self.last_prompt = ""
        
    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self.override_response or "optimized test query"

def test_query_optimizer_empty_history():
    mock_llm = MockLLM()
    optimizer = QueryOptimizer(mock_llm)
    
    # Should bypass LLM if history is empty
    result = optimizer.optimize("what is it?", [])
    assert result == "what is it?"

def test_query_optimizer_with_history():
    mock_llm = MockLLM(override_response="what is node.js?")
    optimizer = QueryOptimizer(mock_llm)
    
    history = [
        {"role": "user", "content": "Tell me about Node"},
        {"role": "assistant", "content": "Node.js is a runtime."}
    ]
    
    result = optimizer.optimize("what is it?", history)
    
    assert result == "what is node.js?"
    assert "Tell me about Node" in mock_llm.last_prompt
    assert "USER QUERY: what is it?" in mock_llm.last_prompt

def test_query_optimizer_strips_quotes():
    mock_llm = MockLLM(override_response='"quoted query"')
    optimizer = QueryOptimizer(mock_llm)
    
    result = optimizer.optimize("query", [{"role": "user", "content": "hi"}])
    assert result == "quoted query"

def test_query_optimizer_fallback():
    # If the LLM generates a massive block of text, fallback to raw query
    long_response = "word " * 40
    mock_llm = MockLLM(override_response=long_response)
    optimizer = QueryOptimizer(mock_llm)
    
    result = optimizer.optimize("short query", [{"role": "user", "content": "hi"}])
    assert result == "short query"
