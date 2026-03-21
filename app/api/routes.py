"""
MODULE: app/api/routes.py

Responsibility:
    HTTP request handling.
    Validates input using Pydantic schemas.
    Enqueues RAG jobs.

Must NOT:
    - Perform RAG logic
    - Call Ollama
    - Query Qdrant
"""

import logging
from fastapi import APIRouter, HTTPException
from rq.job import Job
from app.queue.connection import rag_queue, redis_conn
from app.api.schemas import QueryRequest, QueryResponse, JobStatusResponse
from app.tasks.rag_task import rag_chat_task

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/chat", response_model=QueryResponse)
def chat_endpoint(request: QueryRequest):

    payload = {
        "session_id": request.session_id,
        "query": request.query
    }

    try:
        job = rag_queue.enqueue(rag_chat_task, payload)
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        raise HTTPException(status_code=503, detail="Queue service unavailable. Is Redis running?")

    logger.info(f"Job {job.id} queued for session {request.session_id}")

    return QueryResponse(
        job_id=job.id,
        status="queued"
    )


@router.get("/result/{job_id}", response_model=JobStatusResponse)
def get_result(job_id: str):

    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    if job.is_finished:
        return JobStatusResponse(
            job_id=job_id,
            status="finished",
            result=job.result
        )

    if job.is_failed:
        return JobStatusResponse(
            job_id=job_id,
            status="failed",
            error=str(job.exc_info)
        )

    return JobStatusResponse(
        job_id=job_id,
        status="in_progress"
    )
