"""
routes/documents.py

Endpoints for uploading, listing, and deleting per-chat PDF documents.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.db.deps import get_current_device
from backend.db.models import Chat, Device, Document
from backend.db.session import get_db
from backend.ingestion.pipeline import process_pdf
from backend.ingestion.qdrant_utils import upsert_points, delete_points_by_doc
from backend.models.schemas import DocumentSummary

router = APIRouter()

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB


@router.post(
    "/chats/{chat_id}/documents", response_model=DocumentSummary
)
async def upload_document(
    chat_id: str,
    file: UploadFile,
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> DocumentSummary:
    chat = db.get(Chat, chat_id)
    if chat is None or chat.device_id != device.device_id:
        raise HTTPException(status_code=404, detail="Chat not found")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB",
        )

    # Create DB record first to get the doc_id
    doc = Document(chat_id=chat_id, filename=file.filename)
    db.add(doc)
    db.flush()  # assign doc.document_id

    # Process PDF: extract -> chunk -> embed
    points, page_count = process_pdf(
        pdf_bytes=contents,
        doc_id=doc.document_id,
        chat_id=chat_id,
        filename=file.filename,
    )

    # Index into Qdrant
    upsert_points(points)

    # Update doc record with counts
    doc.page_count = page_count
    doc.chunk_count = len(points)
    db.commit()
    db.refresh(doc)

    return DocumentSummary.model_validate(doc)


@router.get(
    "/chats/{chat_id}/documents", response_model=list[DocumentSummary]
)
def list_documents(
    chat_id: str,
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> list[DocumentSummary]:
    chat = db.get(Chat, chat_id)
    if chat is None or chat.device_id != device.device_id:
        raise HTTPException(status_code=404, detail="Chat not found")

    docs = (
        db.query(Document)
        .filter(Document.chat_id == chat_id)
        .order_by(Document.uploaded_at)
        .all()
    )
    return [DocumentSummary.model_validate(d) for d in docs]


@router.delete(
    "/chats/{chat_id}/documents/{doc_id}", status_code=204
)
def delete_document(
    chat_id: str,
    doc_id: str,
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> None:
    chat = db.get(Chat, chat_id)
    if chat is None or chat.device_id != device.device_id:
        raise HTTPException(status_code=404, detail="Chat not found")

    doc = db.get(Document, doc_id)
    if doc is None or doc.chat_id != chat_id:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from Qdrant first
    delete_points_by_doc(doc_id, chat_id)

    # Delete from DB
    db.delete(doc)
    db.commit()
