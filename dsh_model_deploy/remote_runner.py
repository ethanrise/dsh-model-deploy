from __future__ import annotations

import argparse
import json
import platform
import sys

from benchmark import benchmark_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--provider")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    result = benchmark_onnx(args.model, args.provider, args.warmup, args.runs)
    result["remote"] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
