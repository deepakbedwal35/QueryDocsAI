"""
observability/metrics.py

Two things missing from the app entirely before this: latency
percentiles and per-request cost. Both are cheap to add without a
metrics backend (Prometheus/Datadog) — an in-process rolling window is
enough to answer "are we hitting the PRD's 5-8s target" and "what is
this costing us," and it's a five-minute swap to a real backend later
since the recording call sites (`record_request`) don't change.

Not thread-safe beyond what a single Python process doing async I/O
needs; if you run multiple worker processes (gunicorn -w N) each will
have its own window, which is fine for local dev/small deployments but
should move to a shared store (Redis) before that matters.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from backend.config import GROQ_PRICING_PER_1M

_WINDOW_SIZE = 1000  # keep the last N requests for percentile calc

_lock = threading.Lock()
_latencies_ms: deque[float] = deque(maxlen=_WINDOW_SIZE)
_costs_usd: deque[float] = deque(maxlen=_WINDOW_SIZE)
_request_count = 0
_error_count = 0
_flagged_count = 0


@dataclass
class RequestOutcome:
    total_ms: float
    cost_usd: float
    errored: bool = False
    flagged: bool = False


def estimate_cost_usd(
    prompt_tokens: int, completion_tokens: int, model: str
) -> float:
    pricing = GROQ_PRICING_PER_1M.get(model)
    if pricing is None:
        return 0.0
    return (prompt_tokens / 1_000_000) * pricing["input"] + (
        completion_tokens / 1_000_000
    ) * pricing["output"]


def record_request(outcome: RequestOutcome) -> None:
    global _request_count, _error_count, _flagged_count
    with _lock:
        _latencies_ms.append(outcome.total_ms)
        _costs_usd.append(outcome.cost_usd)
        _request_count += 1
        if outcome.errored:
            _error_count += 1
        if outcome.flagged:
            _flagged_count += 1


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return round(ordered[idx], 2)


def get_snapshot() -> dict:
    with _lock:
        latencies = list(_latencies_ms)
        costs = list(_costs_usd)
        total = _request_count
        errors = _error_count
        flagged = _flagged_count

    return {
        "window_size": len(latencies),
        "total_requests": total,
        "error_count": errors,
        "flagged_count": flagged,
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": round(max(latencies), 2) if latencies else None,
        },
        "cost_usd": {
            "total_in_window": round(sum(costs), 6) if costs else 0.0,
            "avg_per_request": round(sum(costs) / len(costs), 6) if costs else 0.0,
        },
    }
