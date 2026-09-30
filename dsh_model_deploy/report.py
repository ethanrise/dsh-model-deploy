from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_markdown(data: dict[str, Any]) -> str:
    kind = data.get("kind")
    if kind == "comparison":
        lines = ["# Model deployment comparison", "", "| Model | Provider | Avg ms | P95 ms | FPS | Size MB |", "|---|---|---:|---:|---:|---:|"]
        for item in data["results"]:
            lines.append(f"| {Path(item['model']).name} | {item['provider']} | {item['avg_ms']} | {item['p95_ms']} | {item['fps']} | {item['file_size_mb']} |")
        return "\n".join(lines) + "\n"

    bench = data.get("benchmark", data)
    metrics = bench.get("metrics", {})
    lines = [
        "# Model deployment benchmark", "",
        f"- Model: `{Path(bench.get('model', 'unknown')).name}`",
        f"- Target: `{bench.get('target', 'local')}`",
        f"- Provider: `{bench.get('provider', 'unknown')}`",
        f"- Average latency: {metrics.get('avg_ms', 'n/a')} ms",
        f"- P95 latency: {metrics.get('p95_ms', 'n/a')} ms",
        f"- FPS: {metrics.get('fps', 'n/a')}",
    ]
    gate = data.get("gate")
    if gate:
        lines += ["", f"## Deployment Gate: {gate['status']}"]
        for check in gate["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(f"- {mark}: {check['metric']} {check['operator']} {check['target']} (actual: {check['actual']})")
    return "\n".join(lines) + "\n"


def write_report(data: dict[str, Any], path: str) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".md", ".markdown"}:
        output.write_text(render_markdown(data), encoding="utf-8")
    else:
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
