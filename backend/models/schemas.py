"""
schemas.py

Pydantic request/response models for the API, matching the contract
in PRD Section 3 (API Design).
"""

from __future__ import annotations

from pydantic import BaseModel


class AskRequest(BaseModel):
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
