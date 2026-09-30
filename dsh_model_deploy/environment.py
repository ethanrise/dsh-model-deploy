from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import psutil


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_environment() -> dict[str, Any]:
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in (smi or "").splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory_total_mb": _number(parts[1]), "driver_version": parts[2]})
        elif line.strip():
            gpus.append({"raw": line.strip()})

    ort_version = _version("onnxruntime-gpu") or _version("onnxruntime")
    providers: list[str] = []
    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort
        providers = ort.get_available_providers()

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "nvidia_smi": bool(shutil.which("nvidia-smi")),
        "gpus": gpus,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "onnx": _version("onnx"),
        "onnxruntime": ort_version,
        "ort_providers": providers,
        "tensorrt_python": _version("tensorrt"),
        "ssh": bool(shutil.which("ssh")),
        "scp": bool(shutil.which("scp")),
    }


def _number(value: str) -> float | int | str:
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except ValueError:
        return value
