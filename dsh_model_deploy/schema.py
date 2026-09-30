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
        "available_providers": available_providers,
        "metrics": metrics,
        "inputs": inputs,
        "environment": environment or {},
    }
