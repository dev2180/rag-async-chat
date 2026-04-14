from app.llm.base import BaseLLM

class QueryOptimizer:
    """
    Optimizes a raw user query by taking chat history into account,
    resolving anaphora (e.g. 'what is it' -> 'what is node.js'),
    and producing a dense keyword-rich query strictly meant for vector search.
    """
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def optimize(self, raw_query: str, history: list[dict]) -> str:
        if not history:
            # If no history, the query is likely self-contained, but we can still optimize it for search
            # For simplicity, we just pass the raw query if history is empty, to save LLM tokens.
            return raw_query

        history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in history[-4:]]) # up to last 4 interactions

        prompt = f"""
You are an expert search query generator. 
Given the following chat history and a new user query, rewrite the user query so that it is optimized for vector database retrieval.
Resolve any pronouns to their actual entities.
Only output the optimized query, nothing else. No explanations.

CHAT HISTORY:
{history_text}

USER QUERY: {raw_query}

OPTIMIZED QUERY:
""".strip()

        optimized = self.llm.generate(prompt).strip()
        
        # Fallback if LLM outputs something weird
        if len(optimized.split()) > 30 or not optimized:
            return raw_query
            
        # Strip quotes if the LLM wrapped it
        if optimized.startswith('"') and optimized.endswith('"'):
            optimized = optimized[1:-1]
            
        return optimized
