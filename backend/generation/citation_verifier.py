"""
citation_verifier.py

"""

from __future__ import annotations

import re


_CITATION_PATTERN = re.compile(r"\[chunk:\s*([\w\-]+)\s*\]")

_NOT_FOUND_MESSAGE = "I cannot find the answer in the provided documents."


def verify_citations(answer: str, retrieved_chunks: list[dict]) -> dict:
    """
    Extract citation tags from the answer and cross-check them
    against the chunk_ids that were actually retrieved.

    Args:
        answer: Raw LLM output, potentially containing [chunk: id]
            tags.
        retrieved_chunks: The chunks that were passed to the LLM as
            context (same list used to build the prompt).

    Returns:
        {
            "valid_citations": [chunk_id, ...],       # deduped, in order of first appearance
            "hallucinated_citations": [chunk_id, ...], # deduped, in order of first appearance
            "clean_answer": str,                        # hallucinated tags stripped out
        }
    """
    valid_chunk_ids = {c["chunk_id"] for c in retrieved_chunks if "chunk_id" in c}

    # Short-circuit: model explicitly said it couldn't find the
    # answer. Nothing to verify — pass the message through untouched.
    if answer.strip() == _NOT_FOUND_MESSAGE:
        return {
            "valid_citations": [],
            "hallucinated_citations": [],
            "clean_answer": answer.strip(),
        }

    valid_citations: list[str] = []
    hallucinated_citations: list[str] = []
    seen_valid: set[str] = set()
    seen_hallucinated: set[str] = set()

    def _replace(match: re.Match) -> str:
        chunk_id = match.group(1)
        if chunk_id in valid_chunk_ids:
            if chunk_id not in seen_valid:
                valid_citations.append(chunk_id)
                seen_valid.add(chunk_id)
            # Keep valid citation tags in the answer as-is.
            return match.group(0)
        else:
            if chunk_id not in seen_hallucinated:
                hallucinated_citations.append(chunk_id)
                seen_hallucinated.add(chunk_id)
            
            return ""

    clean_answer = _CITATION_PATTERN.sub(_replace, answer)

    # Collapse any double spaces left behind by stripped tags, and
    # tidy up stray spaces before punctuation (e.g. "claim  ." -> "claim.").
    clean_answer = re.sub(r"[ \t]{2,}", " ", clean_answer)
    clean_answer = re.sub(r"\s+([.,;:])", r"\1", clean_answer)
    clean_answer = clean_answer.strip()

    return {
        "valid_citations": valid_citations,
        "hallucinated_citations": hallucinated_citations,
        "clean_answer": clean_answer,
    }
