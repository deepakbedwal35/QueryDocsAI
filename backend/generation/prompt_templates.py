"""
prompt_templates.py

Owns the system prompt contract (PRD Section 4) and the helper that
turns retrieved chunks into the numbered context block the LLM sees.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY the provided
context chunks. Each chunk has a chunk_id.

Rules:
1. Answer only from the given context. Do not use outside knowledge.
2. Every factual claim must include an inline citation in EXACTLY this format: [chunk: chunk_id]
   For example: [chunk: 1705.04742_c2]
   Do NOT use any other format (no "chunk_id:", no brackets around just the id, etc.) —
   it must be the literal text "[chunk: " followed by the chunk_id and "]".
3. If the answer is not contained in the provided context, respond exactly:
   "I cannot find the answer in the provided documents."
4. Do not fabricate chunk_ids. Only cite chunk_ids that appear in the context."""


def format_context(chunks: list[dict]) -> str:
    """
    Turns a list of retrieved chunks into a numbered context block
    for the LLM prompt, e.g.:

        [chunk_id: paper027d_chunk014] (Wiersma, 2017, p.4)
        "input that elicits relatively intense emotions is subjected
        to highly sustainable conscious processing..."

    Args:
        chunks: List of chunk dicts matching the retrieval contract
            (chunk_id, text, paper_title, authors, year, page, score).

    Returns:
        A single formatted string, one block per chunk, ready to be
        dropped into the user/context portion of the prompt. Returns
        an empty string if chunks is empty.
    """
    if not chunks:
        return ""

    blocks = []
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id", "unknown_chunk")
        authors = chunk.get("authors") or []
        year = chunk.get("year", "n.d.")
        page = chunk.get("page")
        text = (chunk.get("text") or "").strip()

        # First-author surname (or "et al." shorthand) for a compact
        # human-readable citation label. Falls back gracefully if
        # authors is missing/empty.
        if authors:
            author_label = authors[0]
            if len(authors) > 1:
                author_label += " et al."
        else:
            author_label = "Unknown author"

        page_label = f", p.{page}" if page is not None else ""

        block = (
            f"[chunk: {chunk_id}] ({author_label}, {year}{page_label})\n" f'"{text}"'
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def build_user_message(question: str, chunks: list[dict]) -> str:
    """
    Assembles the full user-turn content: formatted context + the
    actual question, in the shape the LLM should see after the
    system prompt.
    """
    context_block = format_context(chunks)

    if not context_block:
        # No chunks retrieved at all -> let the model still follow
        # rule 3 and say it can't find the answer.
        context_block = "(No context chunks were retrieved for this question.)"

    return f"Context:\n{context_block}\n\n" f"Question: {question}"
