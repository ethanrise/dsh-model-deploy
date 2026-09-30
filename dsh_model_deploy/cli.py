from __future__ import annotations

import argparse
import json
from typing import Any

from .benchmark import benchmark_onnx
from .environment import probe_environment
from .gate import evaluate_gate
from .inspect import inspect_model
from .ssh import benchmark_over_ssh


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsh-model-deploy",
        description="Benchmark AI models against real local or remote deployment targets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = sub.add_parser("inspect", help="Inspect an ONNX model")
    inspect_cmd.add_argument("model")

    sub.add_parser("env", help="Probe the local deployment environment")

    bench = sub.add_parser("bench", help="Benchmark ONNX Runtime locally")
    bench.add_argument("model")
    bench.add_argument("--provider")
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--runs", type=int, default=50)
    bench.add_argument("--min-fps", type=float)
    bench.add_argument("--max-p95-ms", type=float)
    bench.add_argument("--max-model-mb", type=float)

    remote = sub.add_parser("remote-bench", help="Benchmark on a POSIX host over SSH")
    remote.add_argument("target", help="SSH target or ~/.ssh/config host alias")
    remote.add_argument("model")
    remote.add_argument("--provider")
    remote.add_argument("--warmup", type=int, default=10)
    remote.add_argument("--runs", type=int, default=50)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "inspect":
        _print(inspect_model(args.model))
        return
    if args.command == "env":
        _print(probe_environment())
        return
    if args.command == "remote-bench":
        _print(benchmark_over_ssh(
            args.target,
            args.model,
            provider=args.provider,
            warmup=args.warmup,
            runs=args.runs,
        ))
        return

    model = inspect_model(args.model)
    result = benchmark_onnx(
        args.model,
        provider=args.provider,
        warmup=args.warmup,
        runs=args.runs,
    )
    output: dict[str, Any] = {"model": model, "benchmark": result}
    if args.min_fps is not None or args.max_p95_ms is not None or args.max_model_mb is not None:
        output["gate"] = evaluate_gate(
            result,
            min_fps=args.min_fps,
            max_p95_ms=args.max_p95_ms,
            max_model_mb=args.max_model_mb,
            model_size_mb=model["file_size_mb"],
        )
    _print(output)


if __name__ == "__main__":
    main()
