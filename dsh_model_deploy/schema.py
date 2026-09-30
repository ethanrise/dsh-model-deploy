from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"


def benchmark_result(
    *,
    model: str | Path,
    execution: str,
    provider: str,
    metrics: dict[str, Any],
    inputs: list[dict[str, Any]],
    available_providers: list[str],
    target: str | None = None,
    environment: dict[str, Any] | None = None,
    requested_provider: str | None = None,
    active_providers: list[str] | None = None,
    fallback_reason: str | None = None,
    runtime: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "benchmark",
        "measured": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(Path(model).expanduser()),
        "execution": execution,
        "target": target or "local",
        "provider": provider,
        "requested_provider": requested_provider or provider,
        "active_providers": active_providers or [provider],
        "provider_fallback": requested_provider is not None and requested_provider != provider,
        "fallback_reason": fallback_reason,
        "available_providers": available_providers,
        "runtime": runtime or {},
        "warnings": warnings or [],
        "metrics": metrics,
        "inputs": inputs,
        "environment": environment or {},
    }
