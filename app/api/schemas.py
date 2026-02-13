"""
MODULE: app/api/schemas.py

Responsibility:
    Pydantic models for API request/response validation.

Must NOT:
    - Contain logic
    - Call external services
"""

from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """
    Incoming query request from client.
    """
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique chat session identifier",
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query string",
    )


class QueryResponse(BaseModel):
    """
    Response returned when job is accepted.
    """
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    """
    Response returned when querying job status.
    """
    job_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
