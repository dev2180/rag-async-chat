from typing import List, Tuple
from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding
import numpy as np

class SparseEmbedder:
    """Wrapper for fastembed SparseTextEmbedding (BM25 etc)."""
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        self.model = SparseTextEmbedding(model_name=model_name)

    def embed_text(self, text: str) -> Tuple[List[int], List[float]]:
        """Returns (indices, values) for a single document/query."""
        result = list(self.model.embed([text]))[0]
        
        # fastembed returns numpy arrays, we convert them to native python lists for qdrant client
        indices = result.indices.tolist() if isinstance(result.indices, np.ndarray) else list(result.indices)
        values = result.values.tolist() if isinstance(result.values, np.ndarray) else list(result.values)
        
        return indices, values

    def embed_batch(self, texts: List[str]) -> List[Tuple[List[int], List[float]]]:
        """Returns list of (indices, values) for multiple documents."""
        results = self.model.embed(texts)
        
        formatted_results = []
        for r in results:
            indices = r.indices.tolist() if isinstance(r.indices, np.ndarray) else list(r.indices)
            values = r.values.tolist() if isinstance(r.values, np.ndarray) else list(r.values)
            formatted_results.append((indices, values))
            
        return formatted_results
