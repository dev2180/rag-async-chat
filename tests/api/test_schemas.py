import pytest
from app.api.schemas import QueryRequest, QueryResponse, JobStatusResponse


def test_valid_query():
    data = {"session_id": "test-session-1", "query": "What is RAG?"}
    obj = QueryRequest(**data)
    assert obj.query == "What is RAG?"
    assert obj.session_id == "test-session-1"


def test_empty_query_fails():
    data = {"session_id": "test-session-1", "query": ""}
    with pytest.raises(Exception):
        QueryRequest(**data)


def test_missing_query_fails():
    data = {"session_id": "test-session-1"}
    with pytest.raises(Exception):
        QueryRequest(**data)


def test_missing_session_id_fails():
    data = {"query": "What is RAG?"}
    with pytest.raises(Exception):
        QueryRequest(**data)


def test_empty_session_id_fails():
    data = {"session_id": "", "query": "What is RAG?"}
    with pytest.raises(Exception):
        QueryRequest(**data)


def test_query_response():
    resp = QueryResponse(job_id="abc-123", status="queued")
    assert resp.job_id == "abc-123"
    assert resp.status == "queued"


def test_job_status_response_finished():
    resp = JobStatusResponse(
        job_id="abc-123",
        status="finished",
        result="The answer is 42",
    )
    assert resp.result == "The answer is 42"
    assert resp.error is None


def test_job_status_response_failed():
    resp = JobStatusResponse(
        job_id="abc-123",
        status="failed",
        error="Something went wrong",
    )
    assert resp.result is None
    assert resp.error == "Something went wrong"


def test_session_id_max_length():
    data = {"session_id": "x" * 101, "query": "test"}
    with pytest.raises(Exception):
        QueryRequest(**data)


def test_query_max_length():
    data = {"session_id": "test", "query": "x" * 2001}
    with pytest.raises(Exception):
        QueryRequest(**data)
