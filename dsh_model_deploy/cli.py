from __future__ import annotations

import argparse
import json
from typing import Any

from .benchmark import benchmark_onnx, parse_input_shapes
from .compare import compare_models
from .environment import probe_environment
from .gate import evaluate_gate
from .inspect import inspect_model
from .report import write_report
from .ssh import benchmark_over_ssh, probe_ssh_target


def _emit(data: Any, report: str | None = None) -> None:
    if report:
        write_report(data, report)
    print(json.dumps(data, indent=2, ensure_ascii=False))


def _runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--input-shape", action="append", default=[], metavar="NAME=1x3x640x640")
    parser.add_argument("--default-dynamic-dim", type=int, default=1)
    parser.add_argument("--report", help="Write .json or .md report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dsh-model-deploy", description="Benchmark AI models against real local or remote deployment targets.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="Inspect an ONNX model"); p.add_argument("model")
    sub.add_parser("env", help="Probe the local deployment environment")

    p = sub.add_parser("ssh-preflight", help="Check whether a remote target is benchmark-ready"); p.add_argument("target")

    p = sub.add_parser("bench", help="Benchmark ONNX Runtime locally")
    p.add_argument("model"); _runtime_args(p)
    p.add_argument("--min-fps", type=float); p.add_argument("--max-p95-ms", type=float); p.add_argument("--max-model-mb", type=float)
    p.add_argument("--require-provider")

    p = sub.add_parser("remote-bench", help="Benchmark on a POSIX host over SSH")
    p.add_argument("target"); p.add_argument("model"); _runtime_args(p)

    p = sub.add_parser("compare", help="Benchmark and compare two or more models locally")
    p.add_argument("models", nargs="+"); _runtime_args(p)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect": _emit(inspect_model(args.model)); return
    if args.command == "env": _emit(probe_environment()); return
    if args.command == "ssh-preflight": _emit(probe_ssh_target(args.target)); return

    shapes = parse_input_shapes(args.input_shape)
    if args.command == "remote-bench":
        result = benchmark_over_ssh(args.target, args.model, provider=args.provider, warmup=args.warmup, runs=args.runs, input_shapes=shapes, default_dynamic_dim=args.default_dynamic_dim)
        _emit(result, args.report); return
    if args.command == "compare":
        result = compare_models(args.models, provider=args.provider, warmup=args.warmup, runs=args.runs, input_shapes=shapes, default_dynamic_dim=args.default_dynamic_dim)
        _emit(result, args.report); return

    model = inspect_model(args.model)
    result = benchmark_onnx(args.model, provider=args.provider, warmup=args.warmup, runs=args.runs, input_shapes=shapes, default_dynamic_dim=args.default_dynamic_dim)
    output: dict[str, Any] = {"schema_version": "0.1", "kind": "deployment_evaluation", "model": model, "benchmark": result}
    if any(value is not None for value in (args.min_fps, args.max_p95_ms, args.max_model_mb, args.require_provider)):
        output["gate"] = evaluate_gate(result, min_fps=args.min_fps, max_p95_ms=args.max_p95_ms, max_model_mb=args.max_model_mb, model_size_mb=model["file_size_mb"], required_provider=args.require_provider)
    _emit(output, args.report)


if __name__ == "__main__":
    main()
