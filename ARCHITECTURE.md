# 🏛️ RAG-App V2 Architecture Guide

This document outlines the state-of-the-art RAG pipeline implemented in this project. The system is designed for high precision, semantic understanding, and observability.

## 🔄 High-Level Pipeline Flow

`User Query` ⮕ `Optimizer` ⮕ `Retrieval` ⮕ `Reranking` ⮕ `Compression` ⮕ `Generation`

---

## 🛠️ Component Breakdown

### 1. Ingestion & Chunking (`app/ingestion/`)
*   **Adaptive Chunker:** Uses NLTK `sent_tokenize` for sentence-boundary aware splitting.
*   **Protection:** Automatically detects and preserves fenced code blocks (```) to prevent breaking technical syntax.
*   **Overlapping:** Maintains a 1-2 sentence rolling overlap to preserve context between chunks.

### 2. Embedding Layer (`app/embedding/`)
*   **Dense Embeddings:** Powered by `sentence-transformers/all-MiniLM-L6-v2` for deep semantic meaning.
*   **Sparse Embeddings:** Powered by `FastEmbed` (BM25) for high-precision exact keyword matching.

### 3. Query Optimization (`app/rag/query_optimizer.py`)
*   **Anaphora Resolution:** Resolves pronouns (it, they, this) based on the last 4 messages of chat history.
*   **Multi-Query Expansion:** Uses the LLM to generate 3 diverse versions of the query to maximize retrieval coverage.

### 4. Vector Store (`app/vectorstore/`)
*   **Qdrant:** Stores both Dense and Sparse vectors.
*   **RRF Fusion:** Uses Reciprocal Rank Fusion to merge semantic and keyword search results into a single ranked list.

### 5. Post-Retrieval Refinement (`app/rag/`)
*   **CrossEncoder Reranker:** Takes the top-N candidates and re-scores them using an intensive `ms-marco-MiniLM-L-6-v2` model for extreme precision.
*   **Compressor:** Filters chunks that don't satisfy a minimum rerank score or keyword-overlap density.

### 6. Generation & Output (`app/llm/`)
*   **Ollama (Llama 3.2):** Local LLM that synthesizes the final answer using only the verified context chunks.
*   **Citations:** Automatically maps which source PDF and page number contributed to the answer.

---

## 🧠 Observability
We track EVERYTHING:
*   **Latency:** Every stage (Search, Rerank, LLM) is timed and displayed in the CLI.
*   **Thinking Trace:** Every interaction generates a step-by-step logic log in `logs/thinking_process.log`.

---

> [!IMPORTANT]
> To see the **live thinking process** of your latest query, open the file: [thinking_process.log](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/logs/thinking_process.log)
