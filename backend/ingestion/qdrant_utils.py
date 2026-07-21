"""
qdrant_utils.py

Helpers for upserting and deleting uploaded document points in Qdrant.
"""

from __future__ import annotations

from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

from real_retrieval import _get_qdrant_client, QDRANT_COLLECTION


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
