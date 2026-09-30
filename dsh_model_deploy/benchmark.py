from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


ONNX_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}


def _materialize_shape(shape: list[Any], default_dynamic_dim: int = 1) -> list[int]:
    return [
        int(dim) if isinstance(dim, int) and dim > 0 else default_dynamic_dim
        for dim in shape
    ]


def _input_array(meta: Any) -> np.ndarray:
    shape = _materialize_shape(list(meta.shape))
    dtype = ONNX_TO_NUMPY.get(meta.type, np.float32)
    if np.issubdtype(dtype, np.integer):
        return np.zeros(shape, dtype=dtype)
    return np.random.random(shape).astype(dtype)


def benchmark_onnx(
    model_path: str | Path,
    provider: str | None = None,
    warmup: int = 10,
    runs: int = 50,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Install with: pip install 'dsh-model-deploy[runtime]'"
        ) from exc

    available = ort.get_available_providers()
    if provider is None:
        provider = (
            "CUDAExecutionProvider"
            if "CUDAExecutionProvider" in available
            else "CPUExecutionProvider"
        )
    if provider not in available:
        raise RuntimeError(f"Provider {provider!r} unavailable. Available: {available}")

    session = ort.InferenceSession(str(Path(model_path).expanduser()), providers=[provider])
    feeds = {meta.name: _input_array(meta) for meta in session.get_inputs()}

    for _ in range(max(warmup, 0)):
        session.run(None, feeds)

    samples_ms: list[float] = []
    for _ in range(max(runs, 1)):
        start = time.perf_counter()
        session.run(None, feeds)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples_ms)

    def percentile(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[idx]

    avg = statistics.fmean(samples_ms)
    return {
        "model": str(Path(model_path).expanduser().resolve()),
        "provider": provider,
        "available_providers": available,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "avg_ms": round(avg, 3),
        "p50_ms": round(percentile(0.50), 3),
        "p95_ms": round(percentile(0.95), 3),
        "p99_ms": round(percentile(0.99), 3),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
        "fps": round(1000.0 / avg, 3) if avg > 0 else None,
        "inputs": [
            {"name": item.name, "shape": list(item.shape), "type": item.type}
            for item in session.get_inputs()
        ],
    }
