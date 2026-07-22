"""
groq_client.py

Thin wrapper around Groq's chat completion API. Builds the prompt
from the system prompt + formatted chunks + question, sends it, and
returns the raw answer text (citation verification happens
downstream in citation_verifier.py, not here).
"""

from __future__ import annotations

import os
import os


from dotenv import load_dotenv

from groq import Groq

from .prompt_templates import SYSTEM_PROMPT, build_user_message

GROQ_API_KEY="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" 

DEFAULT_MODEL = "llama-3.3-70b-versatile"

_client: Groq | None = None


def _get_client() -> Groq:
    """Lazily construct the Groq client so import-time doesn't require
    GROQ_API_KEY to already be set (useful for tests that mock this
    module out entirely)."""
    global _client
    if _client is None:
        # api_key = os.environ.get("GROQ_API_KEY")
        # if not api_key:
        #     raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def generate_answer(
    question: str,
    chunks: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """
    Generate an answer grounded strictly in the retrieved chunks.

    Args:
        question: The user's natural-language question.
        chunks: Retrieved chunks matching the retrieval contract
            (chunk_id, text, paper_title, authors, year, page, score).
        model: Groq model id to use.
        temperature: Kept at 0 by default for faithfulness/determinism
            — this is a grounded Q&A task, not creative writing.
        max_tokens: Cap on generated response length.

    Returns:
        Raw answer text from the LLM, including any [chunk: id]
        citation tags it produced. Not yet verified — that's
        citation_verifier.verify_citations()'s job.
    """
    client = _get_client()
    user_message = build_user_message(question, chunks)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content or ""

