from __future__ import annotations

import os
import platform
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .schema import benchmark_result
except ImportError:  # copied to a remote target and run as a plain script
    from schema import benchmark_result

ONNX_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(bool)": np.bool_,
}


def parse_input_shapes(values: list[str] | None) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid input shape {value!r}; expected NAME=1x3x640x640")
        name, raw = value.split("=", 1)
        dims = [int(item) for item in raw.lower().replace(",", "x").split("x") if item]
        if not name or not dims or any(dim <= 0 for dim in dims):
            raise ValueError(f"Invalid input shape {value!r}")
        result[name] = dims
    return result


def _materialize_shape(shape: list[Any], override: list[int] | None, default_dynamic_dim: int) -> list[int]:
    if override is not None:
        if len(override) != len(shape):
            raise ValueError(f"Input shape override rank {len(override)} does not match model rank {len(shape)}")
        return override
    return [int(dim) if isinstance(dim, int) and dim > 0 else default_dynamic_dim for dim in shape]


def _input_array(meta: Any, override: list[int] | None, default_dynamic_dim: int) -> np.ndarray:
    shape = _materialize_shape(list(meta.shape), override, default_dynamic_dim)
    dtype = ONNX_TO_NUMPY.get(meta.type)
    if dtype is None:
        raise RuntimeError(f"Unsupported ONNX Runtime input type {meta.type!r} for input {meta.name!r}")
    if np.issubdtype(dtype, np.integer) or dtype == np.bool_:
        return np.zeros(shape, dtype=dtype)
    return np.random.default_rng(0).random(shape).astype(dtype)


def _create_session(ort: Any, model: str, provider: str) -> tuple[Any, str]:
    """Create a session while capturing native stderr, where ORT logs provider load failures."""
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        return ort.InferenceSession(model, providers=[provider]), ""
    sys.stderr.flush()
    saved = os.dup(fd)
    with tempfile.TemporaryFile(mode="w+b") as capture:
        os.dup2(capture.fileno(), fd)
        try:
            session = ort.InferenceSession(model, providers=[provider])
        finally:
            sys.stderr.flush()
            os.dup2(saved, fd)
            os.close(saved)
        capture.seek(0)
        log = capture.read().decode("utf-8", "replace")
    if log:
        sys.stderr.write(log)
    return session, log


def _fallback_reason(log: str) -> str | None:
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    lines = [line.strip() for line in log.splitlines() if "error" in line.lower() or "require" in line.lower() or "fail" in line.lower()]
    return "\n".join(lines[-5:]) or None


def benchmark_onnx(
    model_path: str | Path,
    provider: str | None = None,
    warmup: int = 10,
    runs: int = 50,
    input_shapes: dict[str, list[int]] | None = None,
    default_dynamic_dim: int = 1,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed. Install with: pip install 'dsh-model-deploy[runtime]'") from exc

    if warmup < 0 or runs < 1 or default_dynamic_dim < 1:
        raise ValueError("warmup must be >= 0, runs >= 1, and default_dynamic_dim >= 1")

    available = ort.get_available_providers()
    if provider is None:
        provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    if provider not in available:
        raise RuntimeError(f"Provider {provider!r} unavailable. Available: {available}")

    requested_provider = provider
    session, session_log = _create_session(ort, str(Path(model_path).expanduser()), provider)
    # ONNX Runtime silently falls back to CPU when the requested provider fails to
    # initialize (e.g. CUDA listed as available but no working driver). Report the
    # provider the session actually uses, not the one that was requested.
    active_providers = session.get_providers()
    provider = active_providers[0] if active_providers else requested_provider
    overrides = input_shapes or {}
    known_inputs = {meta.name for meta in session.get_inputs()}
    unknown = sorted(set(overrides) - known_inputs)
    if unknown:
        raise ValueError(f"Input shape override refers to unknown inputs: {unknown}")

    feeds = {
        meta.name: _input_array(meta, overrides.get(meta.name), default_dynamic_dim)
        for meta in session.get_inputs()
    }

    for _ in range(warmup):
        session.run(None, feeds)

    samples_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feeds)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    ordered = sorted(samples_ms)
    def percentile(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[idx]

    avg = statistics.fmean(samples_ms)
    metrics = {
        "warmup_runs": warmup,
        "measured_runs": runs,
        "avg_ms": round(avg, 3),
        "p50_ms": round(percentile(0.50), 3),
        "p95_ms": round(percentile(0.95), 3),
        "p99_ms": round(percentile(0.99), 3),
        "min_ms": round(min(samples_ms), 3),
        "max_ms": round(max(samples_ms), 3),
        "fps": round(1000.0 / avg, 3) if avg > 0 else None,
    }
    inputs = [
        {"name": meta.name, "model_shape": list(meta.shape), "benchmark_shape": list(feeds[meta.name].shape), "type": meta.type}
        for meta in session.get_inputs()
    ]
    return benchmark_result(
        model=Path(model_path).expanduser().resolve(),
        execution="local",
        provider=provider,
        metrics=metrics,
        inputs=inputs,
        available_providers=available,
        requested_provider=requested_provider,
        active_providers=active_providers,
        fallback_reason=_fallback_reason(session_log) if provider != requested_provider else None,
        runtime={
            "python": sys.executable,
            "python_version": platform.python_version(),
            "onnxruntime": getattr(ort, "__version__", None),
            "numpy": np.__version__,
        },
    )
