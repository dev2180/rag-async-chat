"""
MODULE: app/vectorstore/qdrant_client.py

Responsibility:
    Handles all interactions with Qdrant vector database.

Must NOT:
    - Build prompts
    - Call LLM
"""

from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue

from uuid import uuid4


DEFAULT_COLLECTION = "documents"


class QdrantVectorStore:

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, vector_dimension: int):
        """
        Create collection if it does not exist.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_dimension,
                    distance=Distance.COSINE,
                ),
            )

    def upsert_vectors(
        self,
        vectors: List[List[float]],
        payloads: Optional[List[dict]] = None,
    ):
        """
        Insert vectors into Qdrant.
        """
        points = []

        for idx, vector in enumerate(vectors):
            payload = payloads[idx] if payloads else {}
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        ):
            """
            Perform similarity search.
            """
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
            )
            return results.points
    
    def collection_exists(self) -> bool:
        existing = [c.name for c in self.client.get_collections().collections]
        return self.collection_name in existing
    
    def get_all_doc_ids(self) -> set[str]:
        """
        Returns all unique document IDs stored in the collection.
        """
        if not self.collection_exists():
            return set()

        scroll_result = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
        )

        points, _ = scroll_result

        doc_ids = set()

        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            if doc_id:
                doc_ids.add(doc_id)

        return doc_ids

    def delete_by_doc_id(self, doc_id: str):
        """
        Deletes all vectors belonging to a document.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
