"""
MODULE: app/rag/compressor.py

Filters out low-relevance chunks from the context window before LLM generation.
Reduces token waste by removing paragraphs that don't meaningfully overlap
with the query terms.
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def compress_context(
    payloads: List[Dict],
    query: str,
    min_keyword_overlap: int = 1,
    score_threshold: float = 0.15,
) -> List[Dict]:
    """
    Filter payloads to keep only chunks that are likely relevant.

    Two-pass filter:
      1. Score threshold — drop anything below a minimum retrieval score.
      2. Keyword overlap — drop chunks sharing zero query keywords.

    Args:
        payloads: Retrieved chunk payloads (must have 'text' and 'score' keys).
        query: The user query (or optimized query).
        min_keyword_overlap: Minimum number of query words that must appear in the chunk.
        score_threshold: Minimum retrieval score to keep a chunk.

    Returns:
        Filtered list of payloads.
    """
    if not payloads:
        return []

    query_words = _extract_keywords(query)

    original_count = len(payloads)
    filtered = []

    for p in payloads:
        score = p.get("score", 0.0)
        text = p.get("text", "").lower()

        # Pass 1: Score filter
        if score < score_threshold:
            continue

        # Pass 2: Keyword overlap
        overlap = sum(1 for w in query_words if w in text)
        if overlap < min_keyword_overlap:
            continue

        filtered.append(p)

    # Safety: always keep at least 1 chunk (the best one) even if everything got filtered
    if not filtered and payloads:
        best = max(payloads, key=lambda p: p.get("score", 0.0))
        filtered = [best]

    removed = original_count - len(filtered)
    if removed > 0:
        logger.info(f"Compressor removed {removed}/{original_count} low-relevance chunks")

    return filtered


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text, ignoring stopwords and short tokens."""
    STOPWORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "don", "now", "it", "its", "what", "which", "who", "whom", "this",
        "that", "these", "those", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "they", "them", "their", "and", "but",
        "or", "if", "while", "about", "up", "tell", "explain", "describe",
    }

    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return {w for w in words if w not in STOPWORDS and len(w) > 2}
