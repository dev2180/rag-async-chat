# RUN INSTRUCTIONS — LOCAL RAG SYSTEM

This document explains how to start and run the system correctly during development.

Read this fully before running anything.

---

# 1️⃣ PREREQUISITES

Ensure the following services are running:

* ✅ Ollama running at [http://localhost:11434](http://localhost:11434)
* ✅ Qdrant running in Docker (port 6333)
* ✅ Valkey running in Docker (port 6379)

Check Docker containers:

```
docker ps
```

You should see containers for:

* qdrant
* valkey

---

# 2️⃣ ACTIVATE VIRTUAL ENVIRONMENT

From project root (rag-app/):

Windows:

```
venv\Scripts\activate
```

Linux / Mac:

```
source venv/bin/activate
```

You must see:

```
(venv)
```

before continuing.

---

# 3️⃣ INSTALL DEPENDENCIES (FIRST TIME ONLY)

```
pip install -r requirements.txt
```

---

# 4️⃣ START THE API SERVER (DEVELOPMENT MODE)

Run:

```
uvicorn main:app --reload --reload-dir app
```

What this does:

* Starts FastAPI server
* Automatically reloads on file changes
* Watches only the app/ directory

Access endpoints at:

```
http://127.0.0.1:8000
```

Health check:

```
http://127.0.0.1:8000/health
```

---

# 5️⃣ START WORKERS (MAX = 3)

Open 3 separate terminals.

In each terminal:

```
venv\Scripts\activate
python app/workers/worker.py
```

IMPORTANT:

* Do NOT exceed 3 workers.
* Workers must be started manually.
* Workers do NOT auto-reload.

---

# 6️⃣ SYSTEM ARCHITECTURE FLOW

Client
→ FastAPI (API Layer)
→ Valkey Queue
→ Worker Pool (max 3)
→ RAG Engine
→ Qdrant + Ollama

---

# 7️⃣ STOPPING THE SYSTEM

Stop API:

```
CTRL + C
```

Stop workers:

```
CTRL + C in each worker terminal
```

Stop Docker services (optional):

```
docker stop qdrant
docker stop valkey
```

---

# 8️⃣ DEVELOPMENT RULES

* Never let API call Ollama directly.
* Never scale workers beyond 3.
* Always use the queue.
* Keep layers isolated.

---

# 9️⃣ TROUBLESHOOTING

If API fails:

* Check venv activation
* Check port 8000 availability

If jobs do not process:

* Confirm workers are running
* Confirm Valkey container is running

If LLM fails:

* Check Ollama is active
* Run: ollama ps

---

This file is the authoritative runtime guide for this project.
Follow it strictly.


I am using this using ollama on docker 
docker exec -it ollama ollama pull qwen2.5:7b-instruct
this is the exact command if you wanna download this code's running model for yourself
however note that running ollama on docker will use cpu by default 
instead you could also use simple ollama on laptop



how to delete redis memory for chat history
run it in python shell
from app.chat.memory import ChatMemory
ChatMemory("test_session").clear()