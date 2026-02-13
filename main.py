"""
ENTRYPOINT: main.py

Responsibility:
    FastAPI application entrypoint.
    Wires routes and starts the HTTP server.

Must NOT:
    - Contain business logic
    - Call LLMs
    - Access vector databases directly
"""

from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="RAG Chat API",
    version="1.0.0"
)

app.include_router(router, prefix="/api")
