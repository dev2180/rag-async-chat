import os
from datetime import datetime

TRACE_FILE = "logs/thinking_process.log"

class TraceLogger:
    """
    Logs the step-by-step thinking and transformation process of the RAG pipeline.
    It captures how inputs mutate at each stage (Query Expansion, Retrieval, Compression, Reranking).
    """
    def __init__(self, query: str):
        self.query = query
        self.steps = []
        
    def add_step(self, title: str, details: str):
        self.steps.append((title, details))
        
    def save(self):
        os.makedirs(os.path.dirname(TRACE_FILE), exist_ok=True)
        with open(TRACE_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"🧠 RAG THINKING TRACE | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"👤 Original Query: {self.query}\n")
            f.write(f"{'='*80}\n\n")
            
            for index, (title, details) in enumerate(self.steps, 1):
                f.write(f"┌{'─'*78}\n")
                f.write(f"│ 🛠️  STEP {index}: {title}\n")
                f.write(f"└{'─'*78}\n")
                f.write(f"{details.strip()}\n\n")
            
            f.write(f"✅ TRACE COMPLETE\n")
