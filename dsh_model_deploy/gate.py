from __future__ import annotations

from typing import Any


def evaluate_gate(
    benchmark: dict[str, Any],
    *,
    min_fps: float | None = None,
    max_p95_ms: float | None = None,
    max_model_mb: float | None = None,
    model_size_mb: float | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    if min_fps is not None:
        actual = benchmark.get("fps")
        checks.append({
            "metric": "fps",
            "actual": actual,
            "target": min_fps,
            "operator": ">=",
            "passed": actual is not None and actual >= min_fps,
        })

    if max_p95_ms is not None:
        actual = benchmark.get("p95_ms")
        checks.append({
            "metric": "p95_ms",
            "actual": actual,
            "target": max_p95_ms,
            "operator": "<=",
            "passed": actual is not None and actual <= max_p95_ms,
        })

    if max_model_mb is not None:
        checks.append({
            "metric": "model_size_mb",
            "actual": model_size_mb,
            "target": max_model_mb,
            "operator": "<=",
            "passed": model_size_mb is not None and model_size_mb <= max_model_mb,
        })

    return {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
    }
