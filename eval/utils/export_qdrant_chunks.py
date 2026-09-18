"""
eval/deepeval_suite/export_qdrant_chunks.py

Dumps every stored chunk from Qdrant to JSON so you can scan them and
hand-pick `retrieval_context` / `expected_output` pairs for
golden_dataset.json (the DeepEval golden set).

This is the Qdrant equivalent of a Chroma "dump all chunks" script —
same goal (get every chunk's id/text/metadata out where you can read
it), different API, because we're on Qdrant, not Chroma:

  - Chroma: store._collection.get(include=["documents","metadatas"])
  - Qdrant: client.scroll(collection_name=..., with_payload=True,
            with_vectors=False) — paginated, since scroll returns a
            page + a next_offset, not everything at once.

Run from the repo root:

    python -m eval.deepeval_suite.export_qdrant_chunks
    python -m eval.deepeval_suite.export_qdrant_chunks --source upload --chat-id <id>

Produces eval/deepeval_suite/chunks_dump.json (gitignored — this is a
scratch file for building the golden dataset, not a build artifact
the app depends on).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from backend.config import QDRANT_COLLECTION, get_qdrant_client

OUT_PATH = Path(__file__).parent / "chunks_dump.json"
PAGE_SIZE = 256


def dump_chunks(source: str | None = None, chat_id: str | None = None) -> list[dict]:
    """
    Args:
        source: optional filter, "corpus" or "upload" (matches the
            `source` field your ingestion pipeline writes into each
            point's payload).
        chat_id: optional filter, only relevant when source="upload" —
            restricts the dump to one chat's uploaded documents.

    Returns:
        List of {"id", "text", "meta"} dicts, one per chunk, sorted so
        chunks from the same paper/chat sit together (easier to scan
        by hand when picking golden context).
    """
    client = get_qdrant_client()

    query_filter = None
    conditions = []
    if source:
        conditions.append({"key": "source", "match": {"value": source}})
    if chat_id:
        conditions.append({"key": "chat_id", "match": {"value": chat_id}})
    if conditions:
        query_filter = {"must": conditions}

    dump: list[dict] = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            limit=PAGE_SIZE,
            offset=next_offset,
        )
        for point in points:
            payload = point.payload or {}
            dump.append(
                {
                    "id": str(point.id),
                    "text": payload.get("text", ""),
                    "meta": {
                        "chunk_id": payload.get("chunk_id"),
                        "paper_title": payload.get("paper_title"),
                        "source": payload.get("source"),
                        "chat_id": payload.get("chat_id"),
                        "page": payload.get("page"),
                    },
                }
            )
        if next_offset is None:
            break

    # Group related chunks together: corpus chunks by paper title, upload
    # chunks by chat_id — mirrors the "sort by session" idea from the
    # Chroma version, adapted to our two-population chunk model.
    dump.sort(
        key=lambda c: (
            c["meta"].get("source") or "",
            c["meta"].get("paper_title") or c["meta"].get("chat_id") or "",
            c["meta"].get("chunk_id") or "",
        )
    )
    return dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump Qdrant chunks for golden-dataset building.")
    parser.add_argument("--source", choices=["corpus", "upload"], default=None)
    parser.add_argument("--chat-id", default=None)
    args = parser.parse_args()

    dump = dump_chunks(source=args.source, chat_id=args.chat_id)

    OUT_PATH.write_text(json.dumps(dump, indent=2, ensure_ascii=False))
    print(f"Dumped {len(dump)} chunks to {OUT_PATH}")

    counts = Counter(c["meta"].get("paper_title") or c["meta"].get("chat_id") or "?" for c in dump)
    for key in sorted(counts):
        print(f"  {key}: {counts[key]} chunks")


if __name__ == "__main__":
    main()
