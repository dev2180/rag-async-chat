import pytest
from app.api.schemas import QueryRequest


def test_valid_query():
    data = {"query": "What is RAG?"}
    obj = QueryRequest(**data)
    assert obj.query == "What is RAG?"


def test_empty_query_fails():
    data = {"query": ""}
    with pytest.raises(Exception):
        QueryRequest(**data)


def test_missing_query_fails():
    data = {}
    with pytest.raises(Exception):
        QueryRequest(**data)
