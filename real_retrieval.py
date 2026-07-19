"""
real_retrieval.py

Real retrieval, replacing mock_retrieval.py. Hybrid search (dense via
Qdrant + sparse via BM25), fused with Reciprocal Rank Fusion, then
each result's chunk_id is resolved back to real paper metadata via
pdf_to_paperid.json + metadata.csv.

Returns the exact same shape mock_retrieval.retrieve() did:
    [{chunk_id, text, paper_title, authors, year, page, score}, ...]
so nothing downstream (ask.py, citation_verifier, eval suite) needs
to change -- only the import line in ask.py swaps from
`from mock_retrieval import retrieve` to `from real_retrieval import retrieve`.

Notes on what's NOT available from the current data pipeline:
  - `page` is always None. Page boundaries were discarded before
    chunking in 02_extract_text.ipynb (all pages were concatenated
    into one blob first), so there's no page number to recover.
"""

from __future__ import annotations

import ast
import json
import os
import pickle
import re
import uuid

import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

DATA_DIR = os.environ.get("DATA_DIR", "data")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
PDF_TO_PAPERID_PATH = os.path.join(DATA_DIR, "pdf_to_paperid.json")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
MAPPING_TABLE_PATH = os.path.join(DATA_DIR, "mapping_table.pkl")

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = (
    "docs_collections"  # the FULL collection, not sample_docs_collections
)

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# BGE requires this exact prefix on queries (not on passages) to hit
# its documented retrieval accuracy -- see notes.md. The original
# running_pipeline.py omitted this; adding it here is a deliberate
# quality improvement, not a bug fix.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

FUSION_K = 60  # RRF constant, matches running_pipeline.py / notes.md
CANDIDATE_POOL_SIZE = 20  # how many each of dense/sparse contribute before fusion

# --- Lazy singletons: the embedding model and indexes are expensive
# to load, so load once per process, not per request. ---
_model: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None
_bm25 = None
_bm25_mapping_table: list[dict] | None = None
_paper_by_id: dict[str, dict] | None = None
_pdf_stem_to_paperid: dict[str, str] | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _qdrant_client


def _get_bm25():
    global _bm25, _bm25_mapping_table
    if _bm25 is None:
        with open(BM25_PATH, "rb") as f:
            _bm25 = pickle.load(f)
        with open(MAPPING_TABLE_PATH, "rb") as f:
            _bm25_mapping_table = pickle.load(f)
    return _bm25, _bm25_mapping_table


def _parse_authors(authors_raw: str) -> list[str]:
    """metadata.csv stores authors as a Python-repr'd list of dicts,
    e.g. "[{'authorId': '123', 'name': 'E. Wiersma'}]" -- not valid
    JSON (single quotes), so use ast.literal_eval, not json.loads."""
    try:
        parsed = ast.literal_eval(authors_raw)
        return [a.get("name", "Unknown") for a in parsed]
    except (ValueError, SyntaxError):
        return []


def _get_paper_by_id() -> dict[str, dict]:
    global _paper_by_id
    if _paper_by_id is None:
        meta = pd.read_csv(METADATA_PATH)
        _paper_by_id = {}
        for _, row in meta.iterrows():
            _paper_by_id[row["paperId"]] = {
                "title": row["title"],
                "authors": _parse_authors(row["authors"]),
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
            }
    return _paper_by_id


def _get_pdf_stem_to_paperid() -> dict[str, str]:
    global _pdf_stem_to_paperid
    if _pdf_stem_to_paperid is None:
        with open(PDF_TO_PAPERID_PATH) as f:
            _pdf_stem_to_paperid = json.load(f)
    return _pdf_stem_to_paperid


def _tokenise(text: str) -> list[str]:
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    return clean_text.split()


def _resolve_metadata(chunk_id: str) -> dict:
    """chunk_id looks like '19data_c0' or '1705.04742_c3' -- strip the
    '_c{n}' suffix to get the pdf stem, then chase
    pdf_stem -> paperId -> {title, authors, year}."""
    pdf_stem = chunk_id.rsplit("_c", 1)[0]
    paperid_map = _get_pdf_stem_to_paperid()
    paper_by_id = _get_paper_by_id()

    paper_id = paperid_map.get(pdf_stem)
    paper = paper_by_id.get(paper_id) if paper_id else None

    if paper is None:
        return {"paper_title": "Unknown paper", "authors": [], "year": 0}

    return {
        "paper_title": paper["title"],
        "authors": paper["authors"],
        "year": paper["year"] or 0,
    }


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid retrieval: dense (Qdrant) + sparse (BM25), fused via RRF,
    with each result's chunk_id resolved back to real paper metadata.
    """
    # --- Dense: Qdrant ---
    model = _get_model()
    query_embedding = model.encode(
        BGE_QUERY_PREFIX + question, normalize_embeddings=True
    )
    client = _get_qdrant_client()
    dense_hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=CANDIDATE_POOL_SIZE,
    ).points

    # Qdrant point ids are uuid5(chunk_id), not the chunk_id itself --
    # we need the mapping table (chunk_id <-> text) to translate back,
    # same trick running_pipeline.py used.
    bm25, mapping_table = _get_bm25()
    uuid_to_chunk = {
        str(uuid.uuid5(uuid.NAMESPACE_DNS, item["chunk_id"])): item
        for item in mapping_table
    }

    dense_ranked_chunk_ids = []
    for point in dense_hits:
        item = uuid_to_chunk.get(str(point.id))
        if item:
            dense_ranked_chunk_ids.append(item["chunk_id"])

    # --- Sparse: BM25 ---
    tokens = _tokenise(question)
    scores = bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    sparse_ranked_chunk_ids = [
        mapping_table[i]["chunk_id"]
        for i in ranked_indices[:CANDIDATE_POOL_SIZE]
        if scores[i] > 0
    ]

    # --- Reciprocal Rank Fusion ---
    fused_scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(dense_ranked_chunk_ids, start=1):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
            FUSION_K + rank
        )
    for rank, chunk_id in enumerate(sparse_ranked_chunk_ids, start=1):
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (
            FUSION_K + rank
        )

    ranked_chunk_ids = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[
        :top_k
    ]

    text_by_chunk_id = {item["chunk_id"]: item["text"] for item in mapping_table}

    results = []
    for chunk_id, score in ranked_chunk_ids:
        meta = _resolve_metadata(chunk_id)
        results.append(
            {
                "chunk_id": chunk_id,
                "text": text_by_chunk_id.get(chunk_id, ""),
                "paper_title": meta["paper_title"],
                "authors": meta["authors"],
                "year": meta["year"],
                "page": None,  # not recoverable from the current pipeline, see module docstring
                "score": round(score, 4),
            }
        )
    return results
