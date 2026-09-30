from __future__ import annotations

import importlib.util
import json
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


def probe_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "nvidia_smi": bool(shutil.which("nvidia-smi")),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }

    smi = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if smi:
        info["gpus"] = [
            {"raw": line.strip()} for line in smi.splitlines() if line.strip()
        ]
    else:
        info["gpus"] = []

    if importlib.util.find_spec("onnxruntime"):
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    else:
        info["onnxruntime"] = None
        info["ort_providers"] = []

    info["ssh"] = bool(shutil.which("ssh"))
    info["scp"] = bool(shutil.which("scp"))
    return info


def environment_json() -> str:
    return json.dumps(probe_environment(), indent=2, ensure_ascii=False)
