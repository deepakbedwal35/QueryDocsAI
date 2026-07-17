"""
routes/ask.py

The main /ask endpoint: retrieve -> generate -> verify -> respond.
Also hosts GET /health.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.generation.citation_verifier import verify_citations
from backend.generation.groq_client import generate_answer
from backend.models.schemas import AskRequest, AskResponse, Citation, HealthResponse

# Swap this import for Person A's real retrieval function once it's
# ready (Step 9) — no other code in this file needs to change, since
# both implementations return the same contract shape.
from mock_retrieval import retrieve

router = APIRouter()

_NOT_FOUND_PHRASE = "cannot find the answer"


def build_citation_list(
    valid_chunk_ids: list[str], chunks: list[dict]
) -> list[Citation]:
    """
    Given the chunk_ids the verifier confirmed as valid, build the
    list of Citation objects for the API response, pulling metadata
    from the originally retrieved chunks.

    Args:
        valid_chunk_ids: chunk_ids that passed verification, in the
            order they first appeared in the answer.
        chunks: The full list of retrieved chunks (for metadata
            lookup).

    Returns:
        List of Citation objects, in the same order as
        valid_chunk_ids. Any chunk_id with no matching chunk (should
        not happen, since valid_chunk_ids is derived from these same
        chunks) is silently skipped rather than raising.
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
def ask(req: AskRequest) -> AskResponse:
    chunks = retrieve(req.question)
    raw_answer = generate_answer(req.question, chunks)
    result = verify_citations(raw_answer, chunks)
    citations = build_citation_list(result["valid_citations"], chunks)
    answer_found = _NOT_FOUND_PHRASE not in result["clean_answer"].lower()

    return AskResponse(
        answer=result["clean_answer"],
        citations=citations,
        answer_found=answer_found,
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
