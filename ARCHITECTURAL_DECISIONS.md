# Architectural Decisions and Rationale

This document outlines the major architectural choices made for the RAG Application, contrasting them with alternatives and explaining why they were chosen to optimize precision, performance, and maintainability.

## 1. Hybrid Retrieval (Dense + Sparse) vs. Dense Only
**Decision:** Implement hybrid search using both Dense Embeddings (SentenceTransformers) and Sparse Embeddings (BM25 via FastEmbed), fused using Reciprocal Rank Fusion (RRF).

*   **Dense Only (Alternative):** Uses only semantic similarity.
    *   **Pros:** Easy to set up, good for conceptual queries.
    *   **Cons:** Fails on exact keyword searches, acronyms, or specific part numbers.
*   **Hybrid Search (Chosen):**
    *   **Pros:** Combines semantic understanding with precise keyword matching. RRF intelligently merges the ranked lists.
    *   **Stats/Impact:** In production, Hybrid retrieval typically completes in **~200-250ms**. It significantly increases retrieval recall for specific entities, solving the typical "RAG hallucination" problem where the correct document wasn't retrieved due to a lack of keyword matching.

## 2. CrossEncoder Reranking vs. Bi-Encoder Only
**Decision:** Add a secondary reranking step using a CrossEncoder (`ms-marco-MiniLM-L-6-v2`) after the initial vector retrieval.

*   **Bi-Encoder Only (Alternative):** Relies solely on the cosine similarity scores from the vector database.
    *   **Pros:** Fast, zero overhead post-retrieval.
    *   **Cons:** Bi-encoders compare independent embeddings and miss nuanced relationships between the query and the document.
*   **CrossEncoder Reranking (Chosen):**
    *   **Pros:** The query and document are passed together through the transformer, allowing self-attention mechanisms to weigh the relevance with extreme precision.
    *   **Stats/Impact:** Adds **~300-350ms** of latency but drastically improves precision. For example, during testing, the top CrossEncoder score was often over **8.0** for highly relevant chunks, while irrelevant chunks scored negatively (e.g., **-1.29 to -10.09**), providing a clear separation that cosine similarity cannot achieve.

## 3. Query Optimization (Multi-Query + Anaphora Resolution) vs. Raw Query
**Decision:** Use the LLM to rewrite the user's query into 3 distinct variations and resolve pronouns based on chat history.

*   **Raw Query (Alternative):** Pass the user's query directly to the vector database.
    *   **Pros:** Instantaneous, zero LLM overhead.
    *   **Cons:** Fails when the user uses pronouns ("what did he do?").
*   **Query Optimizer (Chosen):**
    *   **Pros:** Resolves conversational context into standalone queries and expands vocabulary.
    *   **Stats/Impact:** Introduces significant latency (an extra LLM call taking **~8,000ms** depending on the local machine), but guarantees that conversational flow works. Expanding a single query into 3 variants typically pulls in **5 to 10 unique candidate chunks** from the vector store instead of just 2 or 3, drastically increasing the chance of finding the needle in the haystack.

## 4. Two-Pass Context Compression vs. Naive Top-K
**Decision:** Filter retrieved chunks before passing them to the LLM based on a minimum CrossEncoder score and keyword overlap.

*   **Naive Top-K (Alternative):** Always pass the top N chunks to the LLM.
    *   **Pros:** Simple implementation.
    *   **Cons:** Wastes LLM context window on irrelevant information if the retrieval quality was poor, increasing generation latency and hallucination risk.
*   **Context Compressor (Chosen):**
    *   **Pros:** Dynamically sizes the context window.
    *   **Stats/Impact:** Frequently drops 50-80% of retrieved noise. In a real trace, the compressor **dropped 4 out of 5 candidate chunks** due to low score overlap, passing only 1 highly relevant chunk to the LLM. This cuts token costs and speeds up generation time (LLM generation time drops to **~5,400ms** for short contexts).

## 5. Adaptive NLTK Chunking vs. Fixed-Size Character Slicer
**Decision:** Break documents into chunks respecting sentence boundaries and paragraph breaks, while preserving coded blocks.

*   **Fixed-Size Slicer (Alternative):** Splits text every N characters (e.g., 400 chars) with a fixed character overlap.
    *   **Pros:** Extremely fast and uniform chunk sizes.
    *   **Cons:** Cuts sentences in half, destroying semantic meaning at the boundaries.
*   **Adaptive Chunker (Chosen):**
    *   **Pros:** Uses `nltk.sent_tokenize`. Ensures every chunk contains complete sentences and protects ` ``` ` enclosed code blocks.
    *   **Stats/Impact:** Drastically improves embedding quality. A complete sentence embeds much more accurately than a fragmented one, leading to consistently higher coverage metrics (often **80-100%**) during evaluation.

## 6. Async Task Queue (Redis + RQ) vs. Synchronous API
**Decision:** Offload RAG processing to background workers, returning a `job_id` immediately.

*   **Synchronous API (Alternative):** The HTTP request blocks until the LLM generates the answer.
    *   **Pros:** Simple client architecture.
    *   **Cons:** Prone to HTTP timeouts (total query time can exceed **14+ seconds**).
*   **Job Queue (Chosen):**
    *   **Pros:** Decouples the API from the heavy processing. Prevents timeouts.
    *   **Stats/Impact:** Ensures API endpoints respond in **<50ms** while heavy **14s** processing happens asynchronously in the background.

## 7. Module-Level Singletons in Workers vs. Per-Request Instantiation
**Decision:** Load the SentenceTransformer and CrossEncoder models once globally in the worker process.

*   **Per-Request (Alternative):** Initialize models sequentially inside the task.
    *   **Pros:** Completely stateless tasks.
    *   **Cons:** Massive penalty for model loading times.
*   **Singletons (Chosen):**
    *   **Pros:** Models are loaded into memory once when the worker starts.
    *   **Stats/Impact:** Eliminates **3 to 5 seconds** of model materialization overhead per query, enabling the retrieval phase to run in just **~200ms**.

---

## Architecture Summary Comparison Table

| Component Phase | Chosen Architecture | Alternative Considered | Latency Impact | Quality Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval** | Hybrid (Dense + Sparse BM25 + RRF) | Dense Embeddings Only | ~200-250ms (Very Low) | High (Fixes exact-match & acronym misses) |
| **Reranking** | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) | Bi-Encoder Only | +300ms Penalty | Very High (Pushes irrelevant chunks to negative scores) |
| **Query Pre-Processing** | LLM Query Expansion (3 variants) | Raw User Query | +8,000ms Penalty | High (Pools 5-10 candidates; allows conversational pronouns) |
| **Context Assembly** | Compression (Score + Keyword drop) | Naive Top-K | ~2ms (Negligible) | High (Drops 50-80% noise, saves tokens, speeds up LLM) |
| **Text Ingestion** | Adaptive NLTK Sentence Boundary | Fixed Character Slicer | Minimal overhead | Medium (Preserves code blocks and sentence semantics) |
| **API Design** | Async Worker Queue (Redis) | Synchronous HTTP | Makes API instant | High UX (Prevents 15-second browser timeouts) |
| **Model Loading** | Worker-level Global Singletons | Per-Request Init | Saves ~3,000ms | High (Retrieval drops from 3.5s to 200ms) |
