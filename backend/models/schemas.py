"""
schemas.py

Pydantic request/response models for the API, matching the contract
in PRD Section 3 (API Design).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AskRequest(BaseModel):
    chat_id: str
    question: str


class Citation(BaseModel):
    chunk_id: str
    paper_title: str
    authors: list[str]
    year: int
    page: int | None = None
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    answer_found: bool


class HealthResponse(BaseModel):
    status: str


class ChatCreate(BaseModel):
    title: str = "New chat"


class ChatRename(BaseModel):
    title: str


class ChatSummary(BaseModel):
    chat_id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    message_id: str
    role: str
    content: str
    citations: list[Citation] | None = None
    answer_found: bool | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    page_count: int | None = None
    chunk_count: int
    uploaded_at: datetime

    model_config = {"from_attributes": True}
