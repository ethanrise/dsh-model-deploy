from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmark import benchmark_onnx
from .inspect import inspect_model


def compare_models(
    models: list[str],
    *,
    provider: str | None = None,
    warmup: int = 10,
    runs: int = 50,
    input_shapes: dict[str, list[int]] | None = None,
    default_dynamic_dim: int = 1,
) -> dict[str, Any]:
    if len(models) < 2:
        raise ValueError("compare requires at least two models")
    results = []
    for model in models:
        info = inspect_model(model)
        bench = benchmark_onnx(model, provider, warmup, runs, input_shapes, default_dynamic_dim)
        results.append({
            "model": str(Path(model).expanduser().resolve()),
            "file_size_mb": info["file_size_mb"],
            "provider": bench["provider"],
            **bench["metrics"],
        })
    return {
        "schema_version": "0.1",
        "kind": "comparison",
        "measured": True,
        "provider": provider or results[0]["provider"],
        "results": results,
        "fastest_by_avg_ms": min(results, key=lambda item: item["avg_ms"])["model"],
    }
