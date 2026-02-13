from pathlib import Path

COMMENTS = {
    "main.py": """
ENTRYPOINT: main.py

Responsibility:
    FastAPI application entrypoint.
    Wires routes and starts the HTTP server.

Must NOT:
    - Contain business logic
    - Call LLMs
    - Access vector databases directly
""",

    "app/api/routes.py": """
MODULE: app/api/routes.py

Responsibility:
    HTTP request handling.
    Validates input using Pydantic schemas.
    Enqueues RAG jobs.

Must NOT:
    - Perform RAG logic
    - Call Ollama
    - Query Qdrant
""",

    "app/api/schemas.py": """
MODULE: app/api/schemas.py

Responsibility:
    Pydantic models for API request/response validation.

Must NOT:
    - Contain logic
    - Call external services
""",

    "app/queue/connection.py": """
MODULE: app/queue/connection.py

Responsibility:
    Creates and exposes Valkey/Redis + RQ queue connections.

Must NOT:
    - Contain business logic
    - Execute tasks
""",

    "app/workers/worker.py": """
MODULE: app/workers/worker.py

Responsibility:
    Worker bootstrap.
    Pulls jobs from the queue and executes them.

Must NOT:
    - Accept HTTP requests
    - Contain RAG logic
""",

    "app/tasks/rag_task.py": """
MODULE: app/tasks/rag_task.py

Responsibility:
    Defines the function executed by workers.
    Thin wrapper around RAG engine.

Must NOT:
    - Contain API logic
    - Manage queues
""",

    "app/rag/engine.py": """
MODULE: app/rag/engine.py

Responsibility:
    Orchestrates the RAG pipeline:
    retrieve → prompt → generate → return answer.

Must NOT:
    - Talk to Redis
    - Know about FastAPI
""",

    "app/rag/retriever.py": """
MODULE: app/rag/retriever.py

Responsibility:
    Retrieves relevant documents from vector store.

Must NOT:
    - Call LLM
    - Build prompts
""",

    "app/rag/prompt.py": """
MODULE: app/rag/prompt.py

Responsibility:
    Builds prompts from retrieved context and user query.

Must NOT:
    - Call vector DB
    - Call LLM
""",

    "app/llm/ollama_client.py": """
MODULE: app/llm/ollama_client.py

Responsibility:
    Communicates with Ollama LLM server.
    Sends prompt and returns generation.

Must NOT:
    - Contain RAG logic
    - Talk to Qdrant
""",

    "app/vectorstore/qdrant_client.py": """
MODULE: app/vectorstore/qdrant_client.py

Responsibility:
    Handles all interactions with Qdrant vector database.

Must NOT:
    - Build prompts
    - Call LLM
"""
}


def annotate():
    for relative_path, comment in COMMENTS.items():
        path = Path(relative_path)
        if not path.exists():
            continue

        existing = path.read_text().strip()
        if existing:
            continue  # do not overwrite real code

        path.write_text(f'"""\n{comment.strip()}\n"""\n\n', encoding="utf-8")
        print(f"Annotated: {relative_path}")


if __name__ == "__main__":
    annotate()
