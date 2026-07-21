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
import logging
import os
import pickle
import re
import uuid

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

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


_corpus_payload_migrated = False


def _ensure_corpus_payload() -> None:
    """One-time migration: add source='corpus' payload to all points
    that don't already have a 'source' field, so Qdrant filtering
    works.  Points with source='upload' (from user PDF uploads) are
    left untouched."""
    global _corpus_payload_migrated
    if _corpus_payload_migrated:
        return

    client = _get_qdrant_client()

    offset = None
    migrated = 0
    while True:
        result, offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=500,
            offset=offset,
            with_payload=True,
        )
        if not result:
            break

        ids_without_source = [
            p.id for p in result if not p.payload or "source" not in p.payload
        ]
        if ids_without_source:
            client.set_payload(
                collection_name=QDRANT_COLLECTION,
                payload={"source": "corpus"},
                points=ids_without_source,
            )
            migrated += len(ids_without_source)

        if offset is None:
            break

    log.info("Corpus payload check complete: %d points updated.", migrated)
    _corpus_payload_migrated = True


def _resolve_metadata(chunk_id: str, payload: dict | None = None) -> dict:
    """chunk_id looks like '19data_c0' or '1705.04742_c3' -- strip the
    '_c{n}' suffix to get the pdf stem, then chase
    pdf_stem -> paperId -> {title, authors, year}.

    For uploaded chunks (prefix 'upload_'), resolve from the Qdrant
    payload instead."""
    if chunk_id.startswith("upload_") and payload is not None:
        return {
            "paper_title": payload.get("filename", "Uploaded document"),
            "authors": [],
            "year": 0,
        }

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


def retrieve(question: str, top_k: int = 5, chat_id: str | None = None) -> list[dict]:
    """
    Hybrid retrieval: dense (Qdrant) + sparse (BM25), fused via RRF,
    with each result's chunk_id resolved back to real paper metadata.

    When chat_id is provided, the dense search also includes uploaded
    chunks belonging to that chat (via Qdrant payload filter).
    """
    _ensure_corpus_payload()

    # --- Dense: Qdrant ---
    model = _get_model()
    query_embedding = model.encode(
        BGE_QUERY_PREFIX + question, normalize_embeddings=True
    )
    client = _get_qdrant_client()

    # Build filter: corpus OR this chat's uploads
    search_filter = None
    if chat_id:
        search_filter = Filter(
            should=[
                FieldCondition(key="source", match=MatchValue(value="corpus")),
                FieldCondition(key="chat_id", match=MatchValue(value=chat_id)),
            ]
        )

    dense_hits = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=CANDIDATE_POOL_SIZE,
        query_filter=search_filter,
        with_payload=True,
    ).points

    # Qdrant point ids are uuid5(chunk_id), not the chunk_id itself --
    # we need the mapping table (chunk_id <-> text) to translate back,
    # same trick running_pipeline.py used.
    bm25, mapping_table = _get_bm25()
    uuid_to_chunk = {
        str(uuid.uuid5(uuid.NAMESPACE_DNS, item["chunk_id"])): item
        for item in mapping_table
    }

    dense_ranked = []  # list of (chunk_id, text, payload)
    for point in dense_hits:
        point_uuid = str(point.id)
        # Corpus chunk: look up in mapping table
        item = uuid_to_chunk.get(point_uuid)
        if item:
            dense_ranked.append((item["chunk_id"], item["text"], None))
            continue
        # Non-corpus chunk (uploaded or migrated): extract from Qdrant
        # payload.  We check for chunk_id rather than source=='upload'
        # so that points whose source was overwritten by a prior
        # migration are still recovered.
        payload = point.payload or {}
        if "chunk_id" in payload:
            dense_ranked.append((payload["chunk_id"], payload.get("text", ""), payload))

    dense_ranked_chunk_ids = [cid for cid, _, _ in dense_ranked]

    # --- Sparse: BM25 (corpus only — BM25 pickle is static) ---
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

    ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)

    # Guarantee uploaded chunks are represented: uploaded chunks only
    # get dense scores (BM25 is corpus-only), so RRF systematically
    # underranks them.  After initial ranking, swap in any uploaded
    # chunk that scored higher than the weakest corpus chunk in top_k.
    if chat_id:
        top = list(ranked[:top_k])
        corpus_in = [
            (i, cid, sc)
            for i, (cid, sc) in enumerate(top)
            if not cid.startswith("upload_")
        ]
        for up_cid, up_sc in ranked:
            if not up_cid.startswith("upload_"):
                continue
            if up_sc <= 0:
                break  # remaining are worse
            if any(cid == up_cid for cid, _ in top):
                continue  # already in results
            if not corpus_in:
                break  # no corpus slots to replace
            worst_idx, worst_cid, worst_sc = min(corpus_in, key=lambda x: x[2])
            if up_sc <= worst_sc:
                break  # uploaded chunk isn't better than what's there
            top[worst_idx] = (up_cid, up_sc)
            corpus_in = [(i, c, s) for i, c, s in corpus_in if i != worst_idx]
        ranked_chunk_ids = top[:top_k]
    else:
        ranked_chunk_ids = ranked[:top_k]

    # Build lookup for text and payload from the dense results
    text_by_chunk_id = {cid: text for cid, text, _ in dense_ranked}
    payload_by_chunk_id = {cid: pl for cid, _, pl in dense_ranked}
    # Also include corpus text from mapping table
    for item in mapping_table:
        if item["chunk_id"] not in text_by_chunk_id:
            text_by_chunk_id[item["chunk_id"]] = item["text"]

    results = []
    for chunk_id, score in ranked_chunk_ids:
        payload = payload_by_chunk_id.get(chunk_id)
        meta = _resolve_metadata(chunk_id, payload=payload)
        results.append(
            {
                "chunk_id": chunk_id,
                "text": text_by_chunk_id.get(chunk_id, ""),
                "paper_title": meta["paper_title"],
                "authors": meta["authors"],
                "year": meta["year"],
                "page": None,
                "score": round(score, 4),
            }
        )
    return results
