"""
observability/tracing.py

Structured, per-request tracing for /ask. No external dependency
(Langfuse/Phoenix) required to get the core value — this writes one
JSON line per request with everything needed to answer "what did the
system actually do for this specific request":

  - which chunks were retrieved, with fusion scores
  - generation + retrieval + total latency
  - prompt/completion token counts and estimated cost
  - citation coverage / flagged status from the verifier
  - the final answer

Swapping this for Langfuse/Phoenix later is a matter of replacing
`_emit()`'s body with their SDK call — the trace shape built here
(`RequestTrace`) is already the right unit of export.

Logs go to stdout as JSON (one object per line) by default, which is
picked up as-is by any log aggregator (CloudWatch, Render logs, etc.)
without extra wiring. Set TRACE_LOG_PATH to also append to a file.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

logger = logging.getLogger("querydocsai.trace")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)

_LOG_PATH = os.environ.get("TRACE_LOG_PATH")


@dataclass
class RequestTrace:
    request_id: str
    chat_id: str
    question: str
    retrieved_chunks: list[dict] = field(default_factory=list)
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    verification_ms: float | None = None
    total_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated_cost_usd: float | None = None
    citation_coverage: float | None = None
    flagged: bool | None = None
    hallucinated_citation_count: int | None = None
    answer_found: bool | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Tracer:
    """One instance per request. Usage:

        trace = Tracer(chat_id, question)
        with trace.timed("retrieval"):
            chunks = retrieve(...)
        trace.set_chunks(chunks)
        ...
        trace.finish()
    """

    def __init__(self, chat_id: str, question: str):
        self._trace = RequestTrace(
            request_id=str(uuid.uuid4()), chat_id=chat_id, question=question
        )
        self._start = time.perf_counter()

    @contextmanager
    def timed(self, phase: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            setattr(self._trace, f"{phase}_ms", round(elapsed_ms, 2))

    def set_chunks(self, chunks: list[dict]) -> None:
        # Keep the trace small: chunk_id + score only, not full text —
        # full text is already visible in citations on the response.
        self._trace.retrieved_chunks = [
            {"chunk_id": c.get("chunk_id"), "score": c.get("score")} for c in chunks
        ]

    def set_usage(self, prompt_tokens: int, completion_tokens: int, cost_usd: float) -> None:
        self._trace.prompt_tokens = prompt_tokens
        self._trace.completion_tokens = completion_tokens
        self._trace.estimated_cost_usd = round(cost_usd, 6)

    def set_verification(
        self, citation_coverage: float | None, flagged: bool, hallucinated_count: int
    ) -> None:
        self._trace.citation_coverage = citation_coverage
        self._trace.flagged = flagged
        self._trace.hallucinated_citation_count = hallucinated_count

    def set_answer_found(self, answer_found: bool) -> None:
        self._trace.answer_found = answer_found

    def set_error(self, error: str) -> None:
        self._trace.error = error

    def finish(self) -> RequestTrace:
        self._trace.total_ms = round((time.perf_counter() - self._start) * 1000, 2)
        self._emit()
        return self._trace

    def _emit(self) -> None:
        line = json.dumps(self._trace.to_dict(), default=str)
        logger.info(line)
        if _LOG_PATH:
            try:
                os.makedirs(os.path.dirname(_LOG_PATH) or ".", exist_ok=True)
                with open(_LOG_PATH, "a") as f:
                    f.write(line + "\n")
            except OSError:
                logger.warning("Could not write trace to TRACE_LOG_PATH=%s", _LOG_PATH)
