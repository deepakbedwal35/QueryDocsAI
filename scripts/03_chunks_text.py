"""
scripts/03_chunk_text.py

Splits each extracted .txt file into chunks using
RecursiveCharacterTextSplitter, saving one .jsonl file per paper
into data/chunks_jsonl/.

NOTE: your notebook (03_chunks_text.ipynb) was empty/corrupted --
this logic was actually living inside 02_extract_text.ipynb (cell 7).
Split out here so each pipeline stage has exactly one responsibility,
matching the directory structure.

Can be run directly, or imported:
    from scripts.chunk_text import chunk_text, chunk_all
"""

from __future__ import annotations

import json
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_FOLDER_PATH = "../../data/raw_texts"
JSONL_FOLDER_PATH = "../../data/chunks_jsonl"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def chunk_text(text: str) -> list[str]:
    """Split raw text into chunks. Shared by batch script and upload pipeline."""
    return splitter.split_text(text)


def chunk_all(text_folder: str = TEXT_FOLDER_PATH, jsonl_folder: str = JSONL_FOLDER_PATH) -> None:
    """Batch-process every .txt file into a matching .jsonl of chunks."""
    os.makedirs(jsonl_folder, exist_ok=True)

    for file_name in os.listdir(text_folder):
        if not file_name.endswith(".txt"):
            continue

        text_file_path = os.path.join(text_folder, file_name)
        jsonl_file_name = file_name.replace(".txt", ".jsonl")
        jsonl_file_path = os.path.join(jsonl_folder, jsonl_file_name)

        if os.path.exists(jsonl_file_path):
            print(f"Skipping: {file_name} (already chunked)")
            continue

        try:
            with open(text_file_path, "r", encoding="utf-8") as f:
                text_content = f.read()

            chunks = chunk_text(text_content)
            paper_id = jsonl_file_name.replace(".jsonl", "")

            with open(jsonl_file_path, "w", encoding="utf-8") as jsonl_file:
                for chunk_index, chunk in enumerate(chunks):
                    chunk_payload = {
                        "chunk_id": f"{paper_id}_c{chunk_index}",
                        "text": chunk,
                    }
                    jsonl_file.write(json.dumps(chunk_payload, ensure_ascii=False) + "\n")

            print(f"Successfully processed and saved: {jsonl_file_name}")
        except Exception as e:
            print(f"Severe error chunking {file_name}: {e}")


if __name__ == "__main__":
    chunk_all()