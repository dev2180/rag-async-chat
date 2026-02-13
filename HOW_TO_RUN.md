📘 RAG Chat Application — Setup & Run Guide
1️⃣ System Requirements
Required Software

Python 3.10+

Docker Desktop

Ollama (local installation)

Git (optional but recommended)

2️⃣ Required Services

This system depends on:

Service	Purpose
Valkey (Redis)	Queue + Chat Memory
Qdrant	Vector Database
Ollama (Local)	LLM Inference
3️⃣ Installation Steps
🔹 Step 1 — Clone Project
git clone <your-repo-url>
cd rag-app


Or just unzip project folder.

🔹 Step 2 — Create Virtual Environment
python -m venv venv


Activate:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

🔹 Step 3 — Install Dependencies
pip install -r requirements.txt

4️⃣ Start Required Services
🔹 Start Qdrant (Vector DB)
docker run -d ^
  --name rag-vector-db ^
  -p 6333:6333 ^
  qdrant/qdrant


Verify:

http://localhost:6333

🔹 Start Valkey (Redis)
docker run -d ^
  --name valkey ^
  -p 6379:6379 ^
  valkey/valkey


Verify:

docker ps


You should see both containers running.

🔹 Ensure Ollama Is Running

Install Ollama from:

https://ollama.com

Check version:

ollama --version


Pull required model:

ollama pull qwen2.5:7b-instruct


Check available models:

ollama list


Verify API:

curl http://localhost:11434/api/tags

5️⃣ Ingest PDFs Into Vector Database

Place PDF files into:

data/pdfs/


Run ingestion:

python run_ingestion.py


You should see:

[INFO] Creating embeddings...
[INFO] Upsert complete.

6️⃣ Start Worker (Windows Compatible)

⚠ On Windows, use SimpleWorker:

rq worker rag --worker-class rq.worker.SimpleWorker


You should see:

*** Listening on rag...


Leave this terminal running.

7️⃣ Start FastAPI Server

In another terminal:

uvicorn main:app --reload


You should see:

Uvicorn running on http://127.0.0.1:8000

8️⃣ Test API Using Swagger

Open in browser:

http://127.0.0.1:8000/docs

📡 Available API Endpoints
POST /api/chat

Submit a new query.

Request Body:

{
  "session_id": "abc123",
  "query": "What is Node.js?"
}


Response:

{
  "job_id": "uuid",
  "status": "queued"
}

GET /api/result/{job_id}

Retrieve job status.

Response (in progress):

{
  "job_id": "uuid",
  "status": "in_progress",
  "result": null,
  "error": null
}


Response (finished):

{
  "job_id": "uuid",
  "status": "finished",
  "result": "Answer text...",
  "error": null
}

🔁 Conversational Memory

Memory is stored in Redis

Each session uses session_id

Same session_id → conversation continues

Different session_id → new conversation

⚙ Architecture Overview
Client
  ↓
FastAPI
  ↓
Redis Queue (Valkey)
  ↓
RQ Worker
  ↓
RAG Engine
     ├─ Qdrant (vector search)
     ├─ SentenceTransformer (embeddings)
     └─ Ollama (LLM)
  ↓
Redis Memory

🧪 Optional Command Line Testing

POST request:

curl -X POST http://127.0.0.1:8000/api/chat ^
-H "Content-Type: application/json" ^
-d "{\"session_id\":\"abc123\",\"query\":\"What is Node.js?\"}"


Poll result:

curl http://127.0.0.1:8000/api/result/<job_id>

🛑 Troubleshooting
❌ Job not processed

Make sure worker is running:

rq worker rag --worker-class rq.worker.SimpleWorker

❌ Redis connection error

Ensure Valkey container is running:

docker ps

❌ Ollama 404 error

Check:

curl http://localhost:11434/api/tags

🏁 System Startup Order (Important)

Start Docker (Valkey + Qdrant)

Ensure Ollama is running

Run ingestion (if needed)

Start RQ worker

Start FastAPI server

Test via /docs