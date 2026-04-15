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
        
        try:
            # Strip markdown code blocks if the LLM adds them
            if optimized.startswith("```json"):
                optimized = optimized[7:-3].strip()
            elif optimized.startswith("```"):
                optimized = optimized[3:-3].strip()
            
            queries = json.loads(optimized)
            if isinstance(queries, list) and len(queries) > 0:
                logger.debug(f"Generated multi-queries: {queries}")
                return queries[:3]
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse optimizer output as JSON: {optimized}")
        
        # Fallback
        return [raw_query]


