# 📘 RAG Chat Application — Setup & Run Guide

## 1️⃣ System Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | Required for `list[str]` type hints syntax |
| Docker Desktop | For running Qdrant and Valkey containers |
| Ollama | Local installation — run natively on your machine |
| Git | Optional, for cloning |

---

## 2️⃣ Required Services

This system depends on three external services:

| Service | Purpose | Port |
|---|---|---|
| **Valkey** (Redis) | Job queue + Chat memory | 6379 |
| **Qdrant** | Vector database (hybrid dense+sparse) | 6333 |
| **Ollama** | Local LLM inference | 11434 |

---

## 3️⃣ Installation Steps

### Step 1 — Clone the project
```bash
git clone <your-repo-url>
cd rag-app
```
Or just unzip the project folder.

### Step 2 — Create a virtual environment
```bash
python -m venv venv
```

**Activate it:**

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

## 4️⃣ Start Required Services

### Start Qdrant (Vector DB)
```bash
docker run -d --name rag-vector-db -p 6333:6333 qdrant/qdrant
```
Verify it's running: open [http://localhost:6333](http://localhost:6333) in your browser.

### Start Valkey (Redis)
```bash
docker run -d --name valkey -p 6379:6379 valkey/valkey
```
Verify:
```bash
docker ps
```
You should see both containers listed and running.

### Ensure Ollama is running
Install Ollama from: [https://ollama.com](https://ollama.com)

Pull the required model:
```bash
ollama pull llama3.2:latest
```

Check it's available:
```bash
ollama list
```

Verify the API is responding:
```bash
curl http://localhost:11434/api/tags
```

> [!NOTE]
> The default model is `llama3.2:latest`. You can override it by setting the `OLLAMA_MODEL` environment variable before running.

---

## 5️⃣ Add PDFs

Place your PDF files in:
```
data/pdfs/
```

---

## 6️⃣ Run the CLI (Recommended for Dev/Testing)

The CLI automatically runs ingestion then starts an interactive chat session:

```bash
python cli.py
```

This will:
1. Check all required services (Valkey, Qdrant, Ollama)
2. Load AI models (dense embedder + sparse BM25 + cross-encoder reranker)
3. Run the ingestion pipeline (hash check → chunk → embed → upsert)
4. Start an interactive Q&A session with live latency metrics and citations

**Example session:**
```
You > What is Node.js?
Assistant > Node.js is a JavaScript runtime environment...

📎 Sources:
  [1] nodejs.pdf (chunk 3, relevance: 0.91) — "Node.js is a runtime environment..."

📊 Retrieval: 312ms | Rerank: 124ms | top=0.91 avg=0.72 coverage=100% | 1 sources
⏱️  Rewrite: 843ms | LLM: 2310.1ms | Total: 3600.1ms
```

Type `exit` or `quit` to end the session.

---

## 7️⃣ Run in API Mode (Production)

### Step 1 — Start the RQ Worker

> [!IMPORTANT]
> On **Windows**, the standard RQ worker uses `SimpleWorker` automatically (fork() is not available). The `worker.py` script handles this detection for you.

```bash
python -m app.workers.worker
```

Or using the RQ command directly (Windows compatible):
```bash
rq worker rag --worker-class rq.worker.SimpleWorker
```

You should see:
```
*** Listening on rag...
```
Leave this terminal running.

### Step 2 — Start the FastAPI Server

In a new terminal:
```bash
uvicorn main:app --reload
```

You should see:
```
Uvicorn running on http://127.0.0.1:8000
```

### Step 3 — Test via Swagger UI

Open: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 API Endpoints

### `GET /api/health`
Health check — returns `{"status": "ok"}`.

### `POST /api/chat`
Submit a query. Returns a `job_id` immediately (async).

**Request:**
```json
{
  "session_id": "abc123",
  "query": "What is Node.js?"
}
```
**Response:**
```json
{
  "job_id": "some-uuid",
  "status": "queued"
}
```

### `GET /api/result/{job_id}`
Poll for the result.

**Response (finished):**
```json
{
  "job_id": "some-uuid",
  "status": "finished",
  "result": "Node.js is a JavaScript runtime...",
  "error": null
}
```

**Response (in progress):**
```json
{
  "job_id": "some-uuid",
  "status": "in_progress",
  "result": null,
  "error": null
}
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

> [!NOTE]
> `test_reranker.py` and `test_sparse_embedder.py` download AI models on first run (~80MB). Tests that require Redis or Qdrant connections will fail if those services are not running.

Fast tests only (no model loading, no network):
```bash
pytest tests/test_imports.py tests/test_config.py tests/test_latency.py tests/test_compressor.py tests/test_query_optimizer.py tests/test_eval_citations.py tests/test_chunker.py -v
```

---

## 🔁 Conversational Memory

- Memory is stored in Redis per session
- Same `session_id` → conversation continues with prior context
- Different `session_id` → fresh conversation

The Query Optimizer uses the last 4 messages of history to resolve pronouns and generate contextual query variants.

---

## 🏁 Startup Order (Important)

1. Start Docker containers (Valkey + Qdrant)
2. Ensure Ollama is running and model is pulled
3. Place PDFs in `data/pdfs/`
4. **CLI mode:** `python cli.py` (handles ingestion + chat in one command)
5. **API mode:** Start RQ worker → Start FastAPI server → POST to `/api/chat`

---

## 🛑 Troubleshooting

### ❌ "Valkey (Redis) is not running"
```bash
docker run -d --name valkey -p 6379:6379 valkey/valkey
```

### ❌ "Qdrant Vector DB is not running"
```bash
docker run -d --name rag-vector-db -p 6333:6333 qdrant/qdrant
```

### ❌ "Ollama is not running"
Start Ollama, then check: `ollama list` — make sure `llama3.2:latest` is listed.

### ❌ Job not processed
Ensure the RQ worker is running in a separate terminal.

### ❌ No PDFs found
Ensure `.pdf` files are placed inside `data/pdfs/` (not just `data/`).

### ❌ Import errors on startup
Make sure you activated the virtual environment:
```bash
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
```
Then: `pip install -r requirements.txt`