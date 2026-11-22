from __future__ import annotations

import math
from collections import deque
from threading import Lock


class LatencyTracker:
    """In-memory latency tracker for demo /metrics endpoint."""

    def __init__(self, maxlen: int = 500) -> None:
        self._latencies: deque[float] = deque(maxlen=maxlen)
        self._lock = Lock()

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append(latency_ms)

    def snapshot(self) -> dict[str, float | int | None]:
        with self._lock:
            values = list(self._latencies)

        if not values:
            return {
                "request_count": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "last_latency_ms": None,
            }

        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        return {
            "request_count": len(values),
            "avg_latency_ms": round(sum(values) / len(values), 2),
            "p95_latency_ms": round(ordered[idx], 2),
            "last_latency_ms": round(values[-1], 2),
        }


latency_tracker = LatencyTracker()
