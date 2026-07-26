"""
scripts/06_build_bm25_index.py

Builds a single master BM25 index over every chunk in
data/chunks_jsonl/, saving the index + a chunk_id/text mapping table
as pickles for fast loading at query time.

Can be run directly, or imported:
    from scripts.build_bm25_index import build_bm25_index, tokenise
"""

from __future__ import annotations

import json
import os
import pickle
import re

from rank_bm25 import BM25Okapi

CHUNKS_FOLDER_PATH = "../../data/chunks_jsonl"
BM25_FILE_PATH = "../../data/bm25_index.pkl"
MAPPING_FILE_PATH = "../../data/mapping_table.pkl"


def tokenise(text: str) -> list[str]:
    """Lowercase + strip punctuation + split on whitespace.

    Shared by index-building here and by query-time tokenising in
    real_retrieval.py -- keep both in sync if this changes.
    """
    clean_text = re.sub(r"[^\w\s]", " ", text.lower())
    return clean_text.split()


def build_bm25_index(
    chunks_folder: str = CHUNKS_FOLDER_PATH,
    bm25_out: str = BM25_FILE_PATH,
    mapping_out: str = MAPPING_FILE_PATH,
) -> None:
    """Read every chunk across all papers, build one BM25Okapi index,
    and save it + the mapping table (chunk_id -> text) together.

    IMPORTANT: BM25Okapi must be built ONCE over the full corpus, not
    per-file or per-chunk -- it needs the whole corpus to compute
    corpus-wide term stats (IDF, avg doc length).
    """
    corpus: list[list[str]] = []
    mapping_table: list[dict] = []

    for filename in os.listdir(chunks_folder):
        if not filename.endswith(".jsonl"):
            continue

        file_path = os.path.join(chunks_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data_item = json.loads(line)
                chunk_id = data_item["chunk_id"]
                chunk_text = data_item["text"]

                corpus.append(tokenise(chunk_text))
                mapping_table.append({"chunk_id": chunk_id, "text": chunk_text})

    print(f"Building master BM25 index for {len(corpus)} documents...")
    bm25 = BM25Okapi(corpus)
    print("All files processed and loaded successfully!")

    with open(bm25_out, "wb") as f_bm25:
        pickle.dump(bm25, f_bm25)
    with open(mapping_out, "wb") as f_map:
        pickle.dump(mapping_table, f_map)

    print(f"Saved BM25 index to {bm25_out} and mapping table to {mapping_out}")


if __name__ == "__main__":
    build_bm25_index()