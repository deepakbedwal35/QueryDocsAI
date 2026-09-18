"""
scripts/04_build_embeddings.py

Encodes every chunk in data/chunks_jsonl/ using bge-base-en-v1.5,
saving one *_vector.jsonl file per paper into data/vector_embeddings/.

Can be run directly, or imported:
    from scripts.build_embeddings import get_embedding_model, embed_texts, embed_all
"""

from __future__ import annotations

import json
import os

from sentence_transformers import SentenceTransformer

JSONL_FOLDER_PATH = "../../data/chunks_jsonl"
VECTOR_FOLDER_PATH = "../../data/vector_embeddings"

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32

# Lazy singleton -- loading the model is expensive, load once per process
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of chunk texts. Shared by batch script and upload pipeline.

    Note: BGE only needs the "Represent this sentence for searching
    relevant passages:" prefix on QUERIES at retrieval time, not on
    passages/chunks here -- see from backend.retrieval.hybrid import retrieve.
    """
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=min(BATCH_SIZE, len(texts)) or 1,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return [e.tolist() for e in embeddings]


def embed_all(jsonl_folder: str = JSONL_FOLDER_PATH, vector_folder: str = VECTOR_FOLDER_PATH) -> None:
    """Batch-embed every chunks .jsonl file into a matching *_vector.jsonl."""
    os.makedirs(vector_folder, exist_ok=True)

    for file_name in os.listdir(jsonl_folder):
        if not file_name.endswith(".jsonl"):
            continue

        vector_file_name = file_name.replace(".jsonl", "_vector.jsonl")
        vector_file_path = os.path.join(vector_folder, vector_file_name)

        if os.path.exists(vector_file_path):
            print(f"Skipping: {file_name} (already embedded)")
            continue

        jsonl_file_path = os.path.join(jsonl_folder, file_name)

        try:
            chunks = []
            with open(jsonl_file_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))

            texts = [c["text"] for c in chunks]
            vectors = embed_texts(texts)

            with open(vector_file_path, "w", encoding="utf-8") as f_out:
                for chunk, vector in zip(chunks, vectors):
                    payload = {"chunk_id": chunk["chunk_id"], "vector": vector}
                    f_out.write(json.dumps(payload) + "\n")

            print(f"Successfully processed and saved: {vector_file_name}")
        except Exception as e:
            print(f"Severe error embedding {file_name}: {e}")


if __name__ == "__main__":
    embed_all()