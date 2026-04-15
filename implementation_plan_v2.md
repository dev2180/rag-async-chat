# RAG App Enhancement Plan

Add 6 features: **Logging**, **Latency Tracking**, **Retrieval Evaluation**, **Citations**, **Query Optimization**, and **Hybrid Retrieval**.

---

## User Review Required

> [!IMPORTANT]
> **Hybrid retrieval requires re-ingestion.** The Qdrant collection must be recreated with both dense + sparse vector configs. Existing data will need to be re-ingested. This is a one-time migration.

> [!IMPORTANT]
> **Query optimization adds ~1-3s latency per query** because it makes an extra Ollama call to rewrite the query before retrieval. This is a deliberate trade-off: better retrieval quality at the cost of extra time. The CLI will show this as a separate `Rewrite` timing.

> [!WARNING]
> **New dependency: `fastembed`** — Qdrant's own lightweight library for generating sparse (BM25-style) vectors. It's ~15MB, no GPU needed. This replaces the need for a separate tokenizer/BM25 implementation.

---

## Architecture Overview

```
User Query
  │
  ▼
┌─────────────────────┐
│  Query Optimizer     │ ← LLM rewrites query for better retrieval
│  (Ollama call #1)    │
└────────┬────────────┘
         │ optimized_query
         ▼
┌─────────────────────┐
│  Hybrid Retriever    │ ← Dense (SentenceTransformer) + Sparse (FastEmbed BM25)
│  Qdrant Prefetch+RRF │
└────────┬────────────┘
         │ scored results + payloads
         ▼
┌─────────────────────┐
│  Evaluator           │ ← Scores retrieval quality (top-1, avg, coverage)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Prompt Builder      │ ← Includes context chunks + chat history
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  LLM Generation      │ ← Ollama call #2 (the actual answer)
│  (with citations)    │
└────────┬────────────┘
         │ answer + citations + metrics
         ▼
┌─────────────────────┐
│  CLI / API Display   │ ← Shows answer, citations, latency, eval metrics
└─────────────────────┘
```

All timings are captured by the latency tracker and logged to both console and `logs/rag_app.log`.

---

## Proposed Changes

### Phase 1 — Logging Infrastructure

#### [MODIFY] [config.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/config.py)

- Add `LOG_DIR = "logs"` and `LOG_FILE = "rag_app.log"`
- Rewrite `setup_logging()`:
  - **File handler**: `RotatingFileHandler` — 5MB per file, 3 backups, detailed format with timestamps
  - **Console handler**: clean format `[LEVEL] message`, configurable level
  - Accept a `console_level` parameter so CLI can suppress INFO noise

---

### Phase 2 — Latency Tracking

#### [NEW] [latency.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/utils/latency.py)

Provides:
- `LatencyTracker` context manager — wraps any code block and records elapsed time
- `QueryMetrics` dataclass — bundles all per-query timings

```python
@dataclass
class QueryMetrics:
    query_rewrite_ms: float   # Time to optimize query via LLM
    embedding_ms: float       # Time to embed the query
    retrieval_ms: float       # Time for Qdrant hybrid search
    llm_ms: float             # Time for Ollama answer generation
    total_ms: float           # End-to-end
    timestamp: str            # ISO timestamp
```

#### [NEW] [\_\_init\_\_.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/utils/__init__.py)

Empty package init.

---

### Phase 3 — Retrieval Evaluation

#### [NEW] [evaluator.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/rag/evaluator.py)

Takes scored results from Qdrant and computes:

| Metric | What it tells you | Computation |
|---|---|---|
| **Top-1 Score** | Best match relevance | Highest cosine similarity |
| **Avg Score** | Overall retrieval quality | Mean across top-k |
| **Coverage** | % of chunks actually useful | Fraction of results with score ≥ threshold (default `0.3`) |
| **Source Docs** | Which PDFs contributed | Unique `source_name` values from payloads |

Returns `RetrievalMetrics` dataclass:
```python
@dataclass
class RetrievalMetrics:
    top_score: float
    avg_score: float
    coverage: float        # 0.0 to 1.0
    num_results: int
    source_docs: list[str]
```

---

### Phase 4 — Citations

Citations tell the user *exactly where the answer came from*.

#### [NEW] [citations.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/rag/citations.py)

Builds citation objects from retrieval results:

```python
@dataclass
class Citation:
    source: str          # PDF filename (e.g., "nodejs.pdf")
    chunk_index: int     # Which chunk in the document
    relevance: float     # Cosine similarity score
    snippet: str         # First ~150 chars of the chunk text (preview)
```

The module provides:
- `build_citations(scored_results) -> list[Citation]` — extracts citation data from Qdrant results
- `format_citations_cli(citations) -> str` — pretty-prints for terminal
- `citations_to_dict(citations) -> list[dict]` — serializes for API responses

**CLI display example:**
```
📎 Sources:
  [1] nodejs.pdf (chunk 12, relevance: 0.84) — "Node.js is a JavaScript runtime built on..."
  [2] nodejs.pdf (chunk 15, relevance: 0.71) — "The event loop is the core of Node.js..."
  [3] python notes.pdf (chunk 3, relevance: 0.45) — "Unlike Node.js, Python uses threads for..."
```

#### [MODIFY] [prompt.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/rag/prompt.py)

- Update prompt template to instruct the LLM to reference source documents when answering
- Add instruction: *"When possible, mention which source document supports your answer."*

---

### Phase 5 — Query Optimization

Uses Ollama to rewrite the user's raw query into a better retrieval query *before* searching.

#### [NEW] [query_optimizer.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/rag/query_optimizer.py)

```python
class QueryOptimizer:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
    
    def optimize(self, raw_query: str, history: list[dict]) -> str:
        """Rewrites a conversational/vague query into a precise retrieval query."""
```

**How it works:**
1. Sends the raw query + recent chat history to Ollama with a specialized prompt
2. The prompt asks the LLM to:
   - Resolve pronouns/references using chat history (e.g., "tell me more about it" → "explain Node.js event loop in detail")
   - Expand abbreviations and add relevant keywords
   - Keep it concise (one clear search query, not an essay)
3. Returns the optimized query string
4. Falls back to the original query if the LLM call fails

**Optimization prompt template:**
```
You are a search query optimizer. Given the user's question and chat history,
rewrite the question into a clear, keyword-rich search query that will find
the most relevant document chunks.

Rules:
- Resolve pronouns using chat history
- Add relevant technical keywords
- Keep it under 50 words
- Return ONLY the optimized query, nothing else

Chat History: {history}
User Question: {query}
Optimized Query:
```

#### [MODIFY] [config.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/config.py)

- Add `QUERY_OPTIMIZATION_ENABLED = True` — toggle to skip the rewrite step

---

### Phase 6 — Hybrid Retrieval (Dense + Sparse)

This is the biggest change. Currently retrieval is dense-only (SentenceTransformer cosine search). We'll add sparse/BM25-style keyword matching and fuse results using Qdrant's server-side Reciprocal Rank Fusion (RRF).

#### [NEW] [sparse_embedder.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/embedding/sparse_embedder.py)

Uses `fastembed` to generate sparse (BM25-style) vectors:

```python
from fastembed import SparseTextEmbedding

class SparseEmbedder:
    def __init__(self, model_name="Qdrant/bm25"):
        self.model = SparseTextEmbedding(model_name=model_name)
    
    def embed_text(self, text: str) -> tuple[list[int], list[float]]:
        """Returns (indices, values) for sparse vector."""
    
    def embed_batch(self, texts: list[str]) -> list[tuple[list[int], list[float]]]:
        """Batch embed for ingestion."""
```

#### [MODIFY] [qdrant_client.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/vectorstore/qdrant_client.py)

Major changes:
- `create_collection()` — configure **named vectors**: `"dense"` (VectorParams) + `"sparse"` (SparseVectorParams)
- `upsert_vectors()` — accept both dense and sparse vectors per point
- `search()` → `hybrid_search()` — use Qdrant's `prefetch` + `FusionQuery(RRF)`:

```python
def hybrid_search(self, dense_vector, sparse_indices, sparse_values, top_k=5):
    results = self.client.query_points(
        collection_name=self.collection_name,
        prefetch=[
            models.Prefetch(query=dense_vector, using="dense", limit=20),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse", limit=20,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )
    return results.points
```

#### [MODIFY] [retriever.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/rag/retriever.py)

- Accept both `embedder` (dense) and `sparse_embedder` as constructor params
- `retrieve()` generates both dense + sparse vectors from the query
- Calls `hybrid_search()` instead of `search()`
- Returns payloads **and** scored results (for evaluation + citations)

#### [MODIFY] [pipeline.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/ingestion/pipeline.py)

- Accept `sparse_embedder` alongside the existing `embedder`
- During ingestion, generate both dense + sparse vectors for each chunk
- Pass both to `upsert_vectors()`

#### [MODIFY] [requirements.txt](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/requirements.txt)

- Add `fastembed`

---

### Wiring It All Together

#### [MODIFY] [engine.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/rag/engine.py)

The `answer()` method becomes a fully instrumented pipeline:

```python
def answer(self, query, session_id, top_k=5):
    # 1. Get chat history
    memory = ChatMemory(session_id)
    history = memory.get_history()
    
    # 2. Optimize query (timed)
    optimized_query = self.optimizer.optimize(query, history)
    
    # 3. Retrieve with hybrid search (timed)
    results = self.retriever.retrieve(optimized_query, top_k)
    
    # 4. Evaluate retrieval quality
    eval_metrics = evaluate_retrieval(results)
    
    # 5. Build citations
    citations = build_citations(results)
    
    # 6. Build prompt + generate answer (timed)
    prompt = build_prompt(query, context_chunks, history)
    response = self.llm.generate(prompt)
    
    # 7. Save to memory
    memory.add_message("user", query)
    memory.add_message("assistant", response)
    
    # Return everything
    return AnswerResult(answer=response, citations=citations, 
                       metrics=query_metrics, eval=eval_metrics)
```

#### [MODIFY] [cli.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/cli.py)

After each answer, display:

```
Assistant > Node.js is a runtime environment that allows JavaScript
            to run outside the browser...

📎 Sources:
  [1] nodejs.pdf (chunk 12, relevance: 0.84) — "Node.js is a JavaScript runtime..."
  [2] nodejs.pdf (chunk 15, relevance: 0.71) — "The event loop is the core..."

📊 Retrieval: 45ms | top=0.84 avg=0.65 coverage=100% | 2 sources
🔄 Query rewrite: 850ms | "what is node" → "What is Node.js runtime environment"
⏱️  LLM: 3.2s | Total: 4.1s
────────────────────────────────
```

#### [MODIFY] [rag_task.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/tasks/rag_task.py)

- Update to pass sparse embedder to the engine
- Return dict with `answer`, `citations`, and `metrics` from worker

#### [MODIFY] [schemas.py](file:///c:/Users/devsr/OneDrive/Desktop/major%20project/rag-app/app/api/schemas.py)

- Add optional `citations` and `metrics` fields to `JobStatusResponse`

---

## Complete File Summary

| # | Action | File | Purpose |
|---|---|---|---|
| 1 | MODIFY | `app/config.py` | Log config, query optimization toggle |
| 2 | NEW | `app/utils/__init__.py` | Package init |
| 3 | NEW | `app/utils/latency.py` | LatencyTracker + QueryMetrics |
| 4 | NEW | `app/rag/evaluator.py` | Retrieval quality scoring |
| 5 | NEW | `app/rag/citations.py` | Citation extraction + formatting |
| 6 | NEW | `app/rag/query_optimizer.py` | LLM-based query rewriting |
| 7 | NEW | `app/embedding/sparse_embedder.py` | FastEmbed sparse/BM25 vectors |
| 8 | MODIFY | `app/vectorstore/qdrant_client.py` | Hybrid collection + RRF search |
| 9 | MODIFY | `app/rag/retriever.py` | Hybrid retrieve + return scores |
| 10 | MODIFY | `app/rag/prompt.py` | Citation-aware prompt template |
| 11 | MODIFY | `app/rag/engine.py` | Full instrumented pipeline |
| 12 | MODIFY | `app/ingestion/pipeline.py` | Dual-vector ingestion |
| 13 | MODIFY | `cli.py` | Display citations + metrics |
| 14 | MODIFY | `app/tasks/rag_task.py` | Return citations + metrics |
| 15 | MODIFY | `app/api/schemas.py` | Citation + metrics schemas |
| 16 | MODIFY | `requirements.txt` | Add fastembed |

---

## Open Questions

> [!IMPORTANT]
> **Re-ingestion required.** Switching to hybrid retrieval means the Qdrant collection needs to be dropped and recreated with the new schema. Your PDFs will be automatically re-ingested on the next CLI run. Is this acceptable?

> [!NOTE]
> **Query optimization is toggleable.** If you find the extra latency annoying during dev/testing, you can set `QUERY_OPTIMIZATION_ENABLED=False` in config or as an env var to skip it.

---

## Verification Plan

### Automated Tests
- Run existing tests: `venv\Scripts\python.exe -m pytest tests/`
- Verify the collection gets recreated with hybrid config on first run

### Manual Verification
1. Run `python cli.py` and ask 3-4 questions
2. Confirm for each question:
   - ✅ Citations appear below the answer with source PDF names
   - ✅ Query rewrite is shown (original → optimized)
   - ✅ Latency breakdown prints (rewrite, retrieval, LLM, total)
   - ✅ Retrieval eval metrics appear (top-1, avg, coverage)
3. Check `logs/rag_app.log` contains structured entries for every step
4. Test with a vague follow-up like "tell me more about it" and verify the optimizer resolves the pronoun from chat history

## Implementation Order

Build in this exact order (each phase is independently testable):

1. **Logging** → can verify immediately by checking `logs/` directory
2. **Latency tracking** → wrap existing engine, see timing in logs
3. **Retrieval evaluation** → scores appear in logs/CLI
4. **Citations** → visible in CLI output after answers
5. **Query optimization** → visible as "original → rewritten" in CLI
6. **Hybrid retrieval** → requires re-ingestion, biggest change saved for last
