from typing import List, Dict
from dataclasses import dataclass

@dataclass
class Citation:
    source: str
    chunk_index: int
    relevance: float
    snippet: str

def build_citations(payloads: List[Dict]) -> List[Citation]:
    citations = []
    
    for p in payloads:
        source = p.get("source_name", "Unknown Document")
        chunk_index = p.get("chunk_index", -1)
        relevance = p.get("score", 0.0)
        
        text = p.get("text", "")
        # Get first 150 characters, ensuring we don't break mid-word abruptly if possible
        snippet = text[:150].strip()
        if len(text) > 150:
            snippet += "..."
            
        citations.append(Citation(
            source=source,
            chunk_index=chunk_index,
            relevance=relevance,
            snippet=snippet
        ))
        
    return citations

def format_citations_cli(citations: List[Citation]) -> str:
    if not citations:
        return ""
        
    output = "📎 Sources:\n"
    for i, c in enumerate(citations, 1):
        output += f"  [{i}] {c.source} (chunk {c.chunk_index}, relevance: {c.relevance:.2f}) — \"{c.snippet}\"\n"
        
    return output
