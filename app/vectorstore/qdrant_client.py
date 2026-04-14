"""
MODULE: app/vectorstore/qdrant_client.py

Responsibility:
    Handles all interactions with Qdrant vector database.

Must NOT:
    - Build prompts
    - Call LLM
"""

import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector, Prefetch, FusionQuery, Fusion
from qdrant_client.models import Filter, FieldCondition, MatchValue

from uuid import uuid4
from app.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION

logger = logging.getLogger(__name__)


class QdrantVectorStore:

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"QdrantVectorStore connected: {host}:{port}, collection={collection_name}")

    def create_collection(self, vector_dimension: int):
        """
        Create collection with dual vectors (dense and sparse) if it does not exist.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_dimension,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )
            logger.info(f"Created hybrid collection '{self.collection_name}' (dim={vector_dimension})")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def upsert_vectors(
        self,
        dense_vectors: List[List[float]],
        sparse_vectors: List[tuple],
        payloads: Optional[List[dict]] = None,
    ):
        """
        Insert hybrid vectors into Qdrant.
        sparse_vectors should be list of (indices, values)
        """
        points = []

        for idx, d_vec in enumerate(dense_vectors):
            payload = payloads[idx] if payloads else {}
            s_indices, s_values = sparse_vectors[idx]
            
            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector={
                        "dense": d_vec,
                        "sparse": SparseVector(indices=s_indices, values=s_values)
                    },
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Upserted {len(points)} hybrid vectors")

    def search(
        self,
        query_dense: List[float],
        query_sparse: tuple,
        top_k: int = 5,
    ):
        """
        Perform hybrid similarity search using Reciprocal Rank Fusion (RRF).
        """
        s_indices, s_values = query_sparse
        sparse_vec = SparseVector(indices=s_indices, values=s_values)

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(query=query_dense, using="dense", limit=top_k),
                Prefetch(query=sparse_vec, using="sparse", limit=top_k),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
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
        logger.info(f"Deleted vectors for doc_id={doc_id}")
