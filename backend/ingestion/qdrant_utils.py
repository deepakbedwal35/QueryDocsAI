"""
qdrant_utils.py

Helpers for upserting and deleting uploaded document points in Qdrant.
"""

from __future__ import annotations
import os
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = (
    "docs_collections" 
)
_qdrant_client: QdrantClient | None = None

def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def upsert_points(points: list[dict]) -> None:
    """Upsert a batch of points into the docs_collections."""
    client = _get_qdrant_client()
    point_structs = [
        PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=point_structs)


def delete_points_by_doc(doc_id: str, chat_id: str) -> None:
    """Delete all Qdrant points for a specific uploaded document."""
    client = _get_qdrant_client()
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(key="source", match=MatchValue(value="upload")),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="chat_id", match=MatchValue(value=chat_id)),
            ]
        ),
    )
