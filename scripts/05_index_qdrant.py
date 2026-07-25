"""
scripts/05_index_qdrant.py

Pushes every vector in data/vector_embeddings/ into the main Qdrant
collection ("docs_collections"), batched to handle full corpus size.

Can be run directly, or imported:
    from scripts.index_qdrant import get_qdrant_client, upsert_points
"""

from __future__ import annotations

import json
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

VECTOR_FOLDER_PATH = "../../data/vector_embeddings"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "docs_collections"
VECTOR_SIZE = 768  # bge-base-en-v1.5 output dimension
BATCH_SIZE = 100

_client: QdrantClient | None = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection(collection_name: str = COLLECTION_NAME) -> None:
    client = get_qdrant_client()
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection: {collection_name}")


def chunk_id_to_point_id(chunk_id: str) -> str:
    """Qdrant requires int or UUID point ids -- deterministically derive
    a UUID from the chunk_id so re-runs upsert the same point instead
    of duplicating it."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def upsert_points(points: list[dict], collection_name: str = COLLECTION_NAME) -> None:
    """Upsert a list of {id, vector, payload} dicts into Qdrant.

    Shared by the batch script (below) and the live upload pipeline --
    both build points in this same shape before calling this function.
    """
    client = get_qdrant_client()
    ensure_collection(collection_name)

    point_structs = [
        PointStruct(id=p["id"], vector=p["vector"], payload=p.get("payload", {}))
        for p in points
    ]
    client.upsert(collection_name=collection_name, points=point_structs)


def index_all(vector_folder: str = VECTOR_FOLDER_PATH, collection_name: str = COLLECTION_NAME) -> None:
    """Batch-index every vector file in vector_folder into Qdrant.

    Tags every point with source='corpus' so retrieval-time filtering
    (see real_retrieval.py) can distinguish curated corpus chunks from
    user-uploaded ones.
    """
    ensure_collection(collection_name)
    client = get_qdrant_client()

    batch: list[PointStruct] = []
    for file_name in os.listdir(vector_folder):
        if not file_name.endswith(".jsonl"):
            continue

        file_path = os.path.join(vector_folder, file_name)
        print(f"reading file name {file_name}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_item = json.loads(line)
                chunk_id = data_item["chunk_id"]
                vector_data = data_item["vector"]
                point_id = chunk_id_to_point_id(chunk_id)

                batch.append(
                    PointStruct(
                        id=point_id,
                        vector=vector_data,
                        payload={"chunk_id": chunk_id, "source": "corpus"},
                    )
                )

                if len(batch) == BATCH_SIZE:
                    client.upsert(collection_name=collection_name, points=batch)
                    batch = []

    if batch:
        client.upsert(collection_name=collection_name, points=batch)

    print("All files processed and loaded successfully!")


if __name__ == "__main__":
    index_all()