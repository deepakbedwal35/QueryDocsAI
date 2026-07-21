"""
pipeline.py

Synchronous PDF ingestion: extract text -> chunk -> embed -> return
Qdrant-ready points. Mirrors the offline data pipeline but runs at
upload time for user-provided PDFs.
"""

from __future__ import annotations

import io
import uuid

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from real_retrieval import _get_model

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def process_pdf(
    pdf_bytes: bytes, doc_id: str, chat_id: str, filename: str
) -> tuple[list[dict], int]:
    """
    Process an uploaded PDF into Qdrant-ready points.

    Returns:
        (points, page_count) where points is a list of dicts ready
        for Qdrant upsert, and page_count is the number of pages in
        the PDF.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)

    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    chunks = splitter.split_text(full_text)

    model = _get_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)

    points = []
    for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"upload_{doc_id}_c{i}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        points.append(
            {
                "id": point_id,
                "vector": embedding.tolist(),
                "payload": {
                    "source": "upload",
                    "chat_id": chat_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "filename": filename,
                },
            }
        )

    return points, page_count
