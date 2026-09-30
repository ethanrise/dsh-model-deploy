from __future__ import annotations

from typing import Any


def evaluate_gate(
    benchmark: dict[str, Any],
    *,
    min_fps: float | None = None,
    max_p95_ms: float | None = None,
    max_model_mb: float | None = None,
    model_size_mb: float | None = None,
    required_provider: str | None = None,
) -> dict[str, Any]:
    metrics = benchmark.get("metrics", benchmark)
    checks: list[dict[str, Any]] = []

    def add(metric: str, actual: Any, target: Any, operator: str, passed: bool) -> None:
        checks.append({"metric": metric, "actual": actual, "target": target, "operator": operator, "passed": passed})

    if min_fps is not None:
        actual = metrics.get("fps")
        add("fps", actual, min_fps, ">=", actual is not None and actual >= min_fps)
    if max_p95_ms is not None:
        actual = metrics.get("p95_ms")
        add("p95_ms", actual, max_p95_ms, "<=", actual is not None and actual <= max_p95_ms)
    if max_model_mb is not None:
        add("model_size_mb", model_size_mb, max_model_mb, "<=", model_size_mb is not None and model_size_mb <= max_model_mb)
    if required_provider is not None:
        actual = benchmark.get("provider")
        add("provider", actual, required_provider, "==", actual == required_provider)

    return {"status": "PASS" if all(item["passed"] for item in checks) else "FAIL", "checks": checks}
