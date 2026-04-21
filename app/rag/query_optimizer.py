import re
import json
import logging
from typing import List
from app.llm.base import BaseLLM

logger = logging.getLogger(__name__)

class QueryOptimizer:
    """
    Optimizes a raw user query by taking chat history into account,
    resolving anaphora (e.g. 'what is it' -> 'what is node.js'),
    and producing a dense keyword-rich query strictly meant for vector search.
    """
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def optimize(self, raw_query: str, history: list[dict]) -> List[str]:
        if not history:
            prompt = f"""
You are an expert search query generator.
Given a user query, generate 3 different versions of the query to expand the search vocabulary.
Output strictly a JSON list of strings. No explanations, no markdown formatting.

USER QUERY: {raw_query}

JSON OUTPUT:
"""
        else:
            history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-4:]])
            prompt = f"""
You are an expert search query generator. 
Given the following chat history and a new user query, rewrite the user query so that it is optimized for vector database retrieval.
Resolve any pronouns to their actual entities.
Generate up to 3 different versions of the optimized query to expand the search vocabulary.
Output strictly a JSON list of strings. No explanations, no markdown formatting.

CHAT HISTORY:
{history_text}

USER QUERY: {raw_query}

JSON OUTPUT:
"""

        optimized = self.llm.generate(prompt).strip()
        
        # Strip markdown code blocks if the LLM adds them (robust to trailing whitespace/newlines)
        optimized = re.sub(r"^```(?:json)?\s*", "", optimized)
        optimized = re.sub(r"\s*```$", "", optimized).strip()

        # --- Stage 1: Try to parse the entire response as a single JSON array ---
        try:
            queries = json.loads(optimized)
            if isinstance(queries, list) and len(queries) > 0:
                string_queries = [q for q in queries if isinstance(q, str) and q.strip()]
                if string_queries:
                    logger.debug(f"Generated multi-queries (single array): {string_queries}")
                    return string_queries[:3]
        except json.JSONDecodeError:
            pass

        # --- Stage 2: LLM returned multiple separate JSON arrays (one per variant) ---
        # Scan for every [...] block and merge them into one flat list
        found_queries = []
        for match in re.finditer(r"\[.*?\]", optimized, re.DOTALL):
            try:
                chunk = json.loads(match.group(0))
                if isinstance(chunk, list):
                    for item in chunk:
                        if isinstance(item, str) and item.strip() and item not in found_queries:
                            found_queries.append(item)
            except json.JSONDecodeError:
                continue

        if found_queries:
            logger.debug(f"Generated multi-queries (merged arrays): {found_queries[:3]}")
            return found_queries[:3]

        # --- Stage 3: Hard fallback ---
        logger.warning(f"Failed to parse optimizer output as JSON: {optimized!r}")
        return [raw_query]


