from __future__ import annotations

import argparse
import json

from benchmark import benchmark_onnx, parse_input_shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--provider")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--input-shape", action="append", default=[])
    parser.add_argument("--default-dynamic-dim", type=int, default=1)
    args = parser.parse_args()
    result = benchmark_onnx(
        args.model, args.provider, args.warmup, args.runs,
        parse_input_shapes(args.input_shape), args.default_dynamic_dim,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
