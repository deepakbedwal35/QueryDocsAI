"""
routes/ask.py

The main /ask endpoint: retrieve -> generate -> verify -> persist -> respond.
Also hosts GET /health.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.deps import get_current_device
from backend.db.models import Chat, Device, Message
from backend.db.session import get_db, utcnow
from backend.generation.citation_verifier import verify_citations
from backend.generation.groq_client import generate_answer
from backend.models.schemas import AskRequest, AskResponse, Citation, HealthResponse

# Real hybrid retrieval (Qdrant dense + BM25 sparse + RRF fusion),
# resolved back to actual paper metadata. Replaces mock_retrieval now
# that Person A's pipeline + the chunk_id -> paperId mapping are in place.
from real_retrieval import retrieve

router = APIRouter()

_NOT_FOUND_PHRASE = "cannot find the answer"


def build_citation_list(
    valid_chunk_ids: list[str], chunks: list[dict]
) -> list[Citation]:
    """
    Given the chunk_ids the verifier confirmed as valid, build the
    list of Citation objects for the API response, pulling metadata
    from the originally retrieved chunks.
    """
    chunks_by_id = {c["chunk_id"]: c for c in chunks if "chunk_id" in c}

    citations: list[Citation] = []
    for chunk_id in valid_chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        citations.append(
            Citation(
                chunk_id=chunk_id,
                paper_title=chunk.get("paper_title", "Unknown title"),
                authors=chunk.get("authors", []),
                year=chunk.get("year", 0),
                page=chunk.get("page"),
                excerpt=chunk.get("text", ""),
            )
        )
    return citations


@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    device: Device = Depends(get_current_device),
    db: Session = Depends(get_db),
) -> AskResponse:
    chat = db.get(Chat, req.chat_id)
    if chat is None or chat.device_id != device.device_id:
        raise HTTPException(status_code=404, detail="Chat not found")

    chunks = retrieve(req.question)
    raw_answer = generate_answer(req.question, chunks)
    result = verify_citations(raw_answer, chunks)
    citations = build_citation_list(result["valid_citations"], chunks)
    answer_found = _NOT_FOUND_PHRASE not in result["clean_answer"].lower()

    # Persist both turns of the conversation.
    db.add(Message(chat_id=chat.chat_id, role="user", content=req.question))
    db.add(
        Message(
            chat_id=chat.chat_id,
            role="assistant",
            content=result["clean_answer"],
            citations=[c.model_dump() for c in citations],
            answer_found=answer_found,
        )
    )

    # Auto-title the chat from the first question, same behavior the
    # frontend previously did client-side — now it's authoritative
    # server-side so it survives refreshes.
    if chat.title == "New chat":
        trimmed = req.question.strip()
        chat.title = f"{trimmed[:40]}..." if len(trimmed) > 40 else trimmed

    chat.updated_at = utcnow()
    db.commit()

    return AskResponse(
        answer=result["clean_answer"],
        citations=citations,
        answer_found=answer_found,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
