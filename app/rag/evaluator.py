from typing import List, Dict
from dataclasses import dataclass

@dataclass
class RetrievalMetrics:
    top_score: float
    avg_score: float
    coverage: float        # 0.0 to 1.0 (percentage of docs above threshold)
    num_results: int
    source_docs: list[str]

def evaluate_retrieval(payloads: List[Dict], threshold: float = 0.3) -> RetrievalMetrics:
    if not payloads:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0, [])
        
    scores = [p.get("score", 0.0) for p in payloads]
    top_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    above_threshold = sum(1 for s in scores if s >= threshold)
    coverage = above_threshold / len(scores) if scores else 0.0
    
    # Extract unique source names
    sources = set()
    for p in payloads:
        if "source_name" in p:
            sources.add(p["source_name"])
            
    return RetrievalMetrics(
        top_score=top_score,
        avg_score=avg_score,
        coverage=coverage,
        num_results=len(scores),
        source_docs=list(sources)
    )
