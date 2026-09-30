---
name: model-deploy
description: Inspect and benchmark ONNX models against local or SSH-accessible deployment hardware.
---

# Model Deploy

Use this skill when the user wants to determine whether an ONNX model is suitable for a real deployment target.

Prefer measured evidence over estimates.

## Workflow

1. Use `model_inspect` to understand the model before execution.
2. Use `deployment_environment` when benchmarking locally.
3. Use `benchmark_local` for the current machine.
4. Use `benchmark_remote_ssh` when the real target is reachable through an existing SSH configuration.
5. When the user gives explicit requirements such as minimum FPS or maximum P95 latency, pass them to the local benchmark gate.
6. Clearly distinguish static analysis, compatibility observations, and measured benchmark results.

Do not claim performance for hardware that was not actually benchmarked. Do not request or expose SSH passwords or private-key contents; rely on the user's SSH config, ssh-agent, or OS credential handling.
