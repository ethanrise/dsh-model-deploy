from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any


def _require(binary: str) -> None:
    if not shutil.which(binary):
        raise RuntimeError(f"{binary} not found in PATH")


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"Command failed: {args}")
    return result


def probe_ssh_target(target: str) -> dict[str, Any]:
    _require("ssh")
    probe = Path(__file__).resolve().parent / "remote_probe.py"
    command = f"python3 - <<'PY'\n{probe.read_text(encoding='utf-8')}\nPY"
    result = _run(["ssh", target, command], timeout=30)
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    payload["target"] = target
    payload["ready"] = bool(payload.get("modules", {}).get("numpy") and payload.get("modules", {}).get("onnxruntime"))
    return payload


def benchmark_over_ssh(
    target: str,
    model_path: str | Path,
    *,
    provider: str | None = None,
    warmup: int = 10,
    runs: int = 50,
    input_shapes: dict[str, list[int]] | None = None,
    default_dynamic_dim: int = 1,
) -> dict[str, Any]:
    _require("ssh")
    _require("scp")
    preflight = probe_ssh_target(target)
    if not preflight["ready"]:
        raise RuntimeError(f"Remote target is not benchmark-ready: {preflight}")
    if provider and provider not in preflight.get("ort_providers", []):
        raise RuntimeError(f"Remote provider {provider!r} unavailable. Available: {preflight.get('ort_providers', [])}")

    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)

    package_dir = Path(__file__).resolve().parent
    files = [package_dir / name for name in ("benchmark.py", "remote_runner.py", "schema.py")]
    remote_dir = f"/tmp/dsh-model-deploy-{uuid.uuid4().hex[:10]}"
    _run(["ssh", target, f"mkdir -p {shlex.quote(remote_dir)}"])
    try:
        _run(["scp", str(model), *(str(item) for item in files), f"{target}:{remote_dir}/"], timeout=300)
        args = [
            "python3", "remote_runner.py", shlex.quote(f"{remote_dir}/{model.name}"),
            "--warmup", str(warmup), "--runs", str(runs),
            "--default-dynamic-dim", str(default_dynamic_dim),
        ]
        if provider:
            args += ["--provider", shlex.quote(provider)]
        for name, shape in (input_shapes or {}).items():
            args += ["--input-shape", shlex.quote(f"{name}={'x'.join(map(str, shape))}")]
        result = _run(["ssh", target, f"cd {shlex.quote(remote_dir)} && {' '.join(args)}"], timeout=600)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload["execution"] = "ssh"
        payload["target"] = target
        payload["environment"] = preflight
        return payload
    finally:
        subprocess.run(["ssh", target, f"rm -rf {shlex.quote(remote_dir)}"], capture_output=True, text=True, timeout=30, check=False)
