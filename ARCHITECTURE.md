# 🏛️ RAG-App Architecture Guide

This document outlines the production-grade RAG pipeline implemented in this project. The system is designed for high precision, semantic understanding, and observability.

## 🔄 High-Level Pipeline Flow

`User Query` ⮕ `Query Optimizer` ⮕ `Hybrid Retrieval × N` ⮕ `CrossEncoder Reranking` ⮕ `Context Compression` ⮕ `LLM Generation`

---

## 🛠️ Component Breakdown

### 1. Ingestion & Chunking (`app/ingestion/`)
*   **Adaptive Chunker (`chunker.py`):** Uses NLTK `sent_tokenize` for sentence-boundary aware splitting. Respects paragraph breaks (double newlines). Target chunk size: 500 chars.
*   **Code Block Protection:** Automatically detects and preserves fenced code blocks (` ``` `) to prevent breaking technical syntax mid-block.
*   **Sentence-Level Overlap:** Maintains a configurable `overlap_sentences` rolling window (default: 2) to preserve context between consecutive chunks.
*   **Dual Embedding:** Each chunk is embedded twice — dense (SentenceTransformer) and sparse (BM25 via FastEmbed) — and stored in Qdrant as a hybrid vector point.

### 2. Embedding Layer (`app/embedding/`)
*   **Dense Embeddings:** `SentenceTransformerEmbedder` using `all-MiniLM-L6-v2` (384 dimensions). L2-normalized for cosine similarity.
*   **Sparse Embeddings:** `SparseEmbedder` using FastEmbed's BM25 (`Qdrant/bm25`). Returns `(indices, values)` tuples for Qdrant's `SparseVector` format. Captures exact keyword signals dense models miss.

### 3. Query Optimization (`app/rag/query_optimizer.py`)
*   **Anaphora Resolution:** Resolves pronouns (it, they, this) based on the last 4 messages of chat history, so "what about it?" becomes "what about Node.js?".
*   **Multi-Query Expansion:** Uses the LLM to generate up to 3 diverse versions of the query. Each variant is sent through the full retrieval pipeline independently.
*   **Robust Parsing:** Strips LLM markdown code fences (` ```json `) using regex before JSON parsing. Falls back to the original raw query if parsing fails.

### 4. Vector Store (`app/vectorstore/`)
*   **Qdrant:** Stores both Dense and Sparse vectors per chunk in a single hybrid collection.
*   **RRF Fusion:** Uses Qdrant's built-in Reciprocal Rank Fusion (RRF) via `Prefetch` + `FusionQuery` to merge semantic and keyword search results into a single ranked list.
*   **Multi-Query Pooling:** Results from all 3 query variants are pooled and deduplicated by text content before reranking. Pool size capped at `top_k × 2`.

### 5. Post-Retrieval Refinement (`app/rag/`)
*   **CrossEncoder Reranker (`reranker.py`):** Takes the pooled candidate chunks and re-scores each `(query, chunk)` pair using `cross-encoder/ms-marco-MiniLM-L-6-v2` — a dedicated relevance model far more accurate than cosine similarity alone.
*   **Context Compressor (`compressor.py`):** Two-pass filter:
    1. **Score threshold:** drops chunks with cross-encoder score below `0.15`
    2. **Keyword overlap:** drops chunks sharing zero keywords with the (combined) query
    - **Safety fallback:** always keeps at least 1 chunk (the highest-scoring one) even if all fail both filters.

### 6. Generation & Output (`app/llm/`)
*   **Ollama (`llama3.2:latest`):** Local LLM that synthesizes the final answer using **only** the verified context chunks. Configurable via `OLLAMA_MODEL` env var.
*   **Citations (`citations.py`):** Automatically maps which source PDF and chunk index contributed to the answer, with a 150-char snippet per source.
*   **Prompt (`prompt.py`):** System prompt enforces context-only answering, instructs document attribution, and prohibits invented facts.

---

## 🧠 Observability

We track **everything**:

*   **Per-stage Latency (`utils/latency.py`):**  Every pipeline stage (Query Rewrite, Retrieval, Reranking, Compression, LLM) is timed independently using `LatencyTracker` context managers. Metrics collected in `QueryMetrics` dataclass and printed in the CLI after every response.

*   **Thinking Trace (`utils/trace.py`):** Every interaction appends a structured step-by-step log entry to `logs/thinking_process.log` capturing:
    - Generated query variants
    - Number of pooled retrieval candidates
    - Full reranking scores and chunk snippets
    - Compression decisions (how many chunks were dropped)
    - The final LLM response

*   **Retrieval Quality Metrics (`rag/evaluator.py`):** Per-query evaluation of `top_score`, `avg_score`, `coverage` (% of results above threshold), and `source_docs`.

---

> [!IMPORTANT]
> To see the **live thinking process** of your latest query, open the file:
> [thinking_process.log](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/logs/thinking_process.log)

---

## 🧩 Module Responsibility Map

```
app/
├── config.py             ← All configuration constants + setup_logging()
├── api/
│   ├── routes.py         ← HTTP endpoints only (no RAG logic)
│   └── schemas.py        ← Pydantic request/response models
├── chat/
│   └── memory.py         ← Redis-backed session chat history
├── embedding/
│   ├── base.py           ← BaseEmbedder ABC
│   ├── sentence_transformer_embedder.py  ← Dense (384-dim)
│   └── sparse_embedder.py               ← BM25 sparse
├── ingestion/
│   ├── pdf_loader.py     ← PDF text extraction
│   ├── chunker.py        ← Adaptive sentence-aware chunker
│   ├── hasher.py         ← SHA-256 file identity
│   └── pipeline.py       ← Ingestion orchestrator
├── llm/
│   ├── base.py           ← BaseLLM ABC
│   └── ollama_client.py  ← Ollama API client (blocking + streaming)
├── queue/
│   └── connection.py     ← Redis + RQ queue setup
├── rag/
│   ├── engine.py         ← Pipeline orchestrator (answer + stream_answer)
│   ├── retriever.py      ← Hybrid retrieval (dense + sparse)
│   ├── query_optimizer.py← Multi-query expansion + anaphora resolution
│   ├── reranker.py       ← CrossEncoder reranking
│   ├── compressor.py     ← Context filtering / compression
│   ├── prompt.py         ← Prompt assembly
│   ├── evaluator.py      ← Retrieval quality metrics
│   └── citations.py      ← Source attribution
├── tasks/
│   └── rag_task.py       ← RQ worker task + singleton engine init
├── utils/
│   ├── latency.py        ← Stage timing (LatencyTracker, QueryMetrics)
│   └── trace.py          ← Step-by-step decision log
├── vectorstore/
│   └── qdrant_client.py  ← Qdrant hybrid collection wrapper
└── workers/
    └── worker.py         ← Worker bootstrap (auto-detects Windows)
```
