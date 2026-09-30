---
name: model-deploy
description: Inspect and benchmark ONNX models against local or SSH-accessible deployment hardware, with PASS/FAIL deployment gates.
---

# Model Deploy

Use this skill when the user wants to determine whether an ONNX model is suitable for a real deployment target.

Prefer measured evidence over estimates.

## Workflow

1. `model_inspect` first: shapes, opset, operators, size, and whether inputs are dynamic.
2. If `dynamic_inputs` is true, pass `inputShapes` (e.g. `{"images": [1, 3, 640, 640]}`) or `defaultDynamicDim` matching the real deployment batch/resolution. Say which shape was benchmarked.
3. Local target: `deployment_environment`, then `benchmark_local`.
4. Remote target: `ssh_preflight` first; only call `benchmark_remote_ssh` when `ready` is true. If not ready, report what is missing — do not install anything on the target unless the user asks.
5. When the user gives requirements (min FPS, max P95 latency, max model size, required provider), pass them as gate arguments (`minFps`, `maxP95Ms`, `maxModelMb`, `requireProvider`) and report the gate verdict per check.

## Reading results

- `provider` is the execution provider the session **actually** used. If `provider_fallback` is true, say so explicitly, quote `fallback_reason`, and do not present the numbers as GPU/accelerator performance.
- `runtime` records the interpreter and ONNX Runtime version that produced the numbers; mention it when comparing runs.
- If `warnings` is non-empty (low warmup/runs, high `cv`), state that the numbers are unstable and prefer re-running with defaults (warmup 10, runs 50) or more.
- Inputs are synthetic (random floats, zero integers). Latency is meaningful only for models whose cost does not depend on input values; integer inputs that control shapes (e.g. Reshape targets) will fail or be meaningless. This tool never checks accuracy.

Clearly distinguish static analysis, compatibility observations, and measured benchmark results. Do not claim performance for hardware that was not actually benchmarked. Do not request or expose SSH passwords or private-key contents; rely on the user's SSH config, ssh-agent, or OS credential handling.
