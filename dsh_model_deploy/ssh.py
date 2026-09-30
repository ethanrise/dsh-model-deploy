from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import tempfile
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


def benchmark_over_ssh(
    target: str,
    model_path: str | Path,
    *,
    provider: str | None = None,
    warmup: int = 10,
    runs: int = 50,
) -> dict[str, Any]:
    """Run an agentless benchmark on a POSIX SSH target using its existing Python/ORT environment."""
    _require("ssh")
    _require("scp")

    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)

    package_dir = Path(__file__).resolve().parent
    benchmark_py = package_dir / "benchmark.py"
    runner_py = package_dir / "remote_runner.py"
    remote_dir = f"/tmp/dsh-model-deploy-{uuid.uuid4().hex[:10]}"

    _run(["ssh", target, f"mkdir -p {shlex.quote(remote_dir)}"])
    try:
        _run([
            "scp",
            str(model),
            str(benchmark_py),
            str(runner_py),
            f"{target}:{remote_dir}/",
        ], timeout=300)

        remote_model = f"{remote_dir}/{model.name}"
        command = [
            "cd", shlex.quote(remote_dir), "&&",
            "python3", "remote_runner.py", shlex.quote(remote_model),
            "--warmup", str(warmup),
            "--runs", str(runs),
        ]
        if provider:
            command.extend(["--provider", shlex.quote(provider)])

        result = _run(["ssh", target, " ".join(command)], timeout=600)
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        payload["ssh_target"] = target
        return payload
    finally:
        subprocess.run(
            ["ssh", target, f"rm -rf {shlex.quote(remote_dir)}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
