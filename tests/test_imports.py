"""
Tests to verify all module imports work without errors.
This catches circular imports, missing dependencies, and broken references.
"""

import pytest


def test_import_config():
    from app.config import REDIS_HOST, OLLAMA_MODEL, setup_logging


def test_import_schemas():
    from app.api.schemas import QueryRequest, QueryResponse, JobStatusResponse


def test_import_embedding_base():
    from app.embedding.base import BaseEmbedder


def test_import_llm_base():
    from app.llm.base import BaseLLM


def test_import_chunker():
    from app.ingestion.chunker import chunk_text


def test_import_hasher():
    from app.ingestion.hasher import compute_file_hash


def test_import_pdf_loader():
    from app.ingestion.pdf_loader import load_pdf_text


def test_import_prompt():
    from app.rag.prompt import build_prompt


def test_import_prompt_builds():
    """Test that build_prompt runs correctly."""
    from app.rag.prompt import build_prompt

    result = build_prompt(
        query="What is Python?",
        context_chunks=["Python is a language.", "It is interpreted."],
        history=[{"role": "user", "content": "hi"}],
    )
    assert "What is Python?" in result
    assert "Python is a language." in result
    assert "USER: hi" in result
