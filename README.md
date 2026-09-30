# dsh-model-deploy

**Local or remote AI model deployment benchmarking for real hardware targets.**

`dsh-model-deploy` helps answer a practical deployment question: **will this model meet my requirements on the machine that will actually run it?**

V0.1 focuses on ONNX + ONNX Runtime, with local execution on Windows/Linux and agentless remote execution over SSH to POSIX targets such as Linux workstations and Jetson devices.

## V0.1 capabilities

- ONNX static inspection: inputs/outputs, shapes, opset, parameter count, operators, model size and dynamic-input detection
- Deployment environment probe: OS, CPU, RAM, NVIDIA GPU, ONNX Runtime providers, SSH/SCP
- Local ONNX Runtime benchmark: warmup, average latency, P50/P95/P99, FPS
- Agentless SSH benchmark: copy a lightweight runner + model to a temporary remote directory, execute, collect JSON, clean up
- Deployment Gate: validate measured FPS, P95 latency and model-size constraints
- DSH/MCP tools for model inspection, environment probing, local benchmark and remote SSH benchmark

## Install

### 1. Python runtime

The plugin runs its benchmarks in Python. Any interpreter with these packages works:

```bash
pip install numpy onnx onnxruntime psutil
```

For CPU inference, standard `onnxruntime` is enough. GPU execution requires an ONNX Runtime build/provider compatible with the machine (e.g. `onnxruntime-gpu` with matching CUDA/cuDNN libraries); if the provider fails to load, results say so rather than reporting CPU numbers as GPU numbers.

The plugin picks the interpreter automatically: `DSH_MODEL_DEPLOY_PYTHON` if set, then `python3`/`python`, then conda/miniforge environments under your home directory — the first one that can import `numpy`, `onnx` and `onnxruntime`.

### 2. DSH plugin

In DeepSeek Harness, open **Plugins → Add plugin** and enter:

```
https://github.com/ethanrise/dsh-model-deploy
```

(or a local checkout path). This registers the `model-deploy` MCP server and the `model-deploy` skill; no `npm link` or PATH setup is needed. Restart DSH after installing or updating.

### Standalone CLI (optional)

```bash
git clone https://github.com/ethanrise/dsh-model-deploy.git
cd dsh-model-deploy
pip install -e ".[runtime]"
npm install && npm run build   # only needed after changing src/
node lib/doctor.js             # shows which interpreter was chosen and checks dependencies
```

## CLI

Inspect a model:

```bash
dsh-model-deploy inspect model.onnx
```

Probe the current machine:

```bash
dsh-model-deploy env
```

Benchmark locally:

```bash
dsh-model-deploy bench model.onnx --runs 100
```

Benchmark with deployment constraints:

```bash
dsh-model-deploy bench model.onnx \
  --min-fps 30 \
  --max-p95-ms 35 \
  --max-model-mb 200
```

Benchmark a real remote target using an existing SSH alias:

```bash
dsh-model-deploy remote-bench orin-dev model.onnx --runs 100
```

The remote target currently needs `python3`, `numpy`, and `onnxruntime` already installed. V0.1 deliberately **detects/uses the environment rather than modifying it**.

## DSH tools

- `model_inspect`
- `deployment_environment`
- `benchmark_local`
- `ssh_preflight`
- `benchmark_remote_ssh`

Model paths must be absolute (or start with `~/`).

`benchmark_local` and `benchmark_remote_ssh` accept `inputShapes` (e.g. `{"images": [1, 3, 640, 640]}`) and `defaultDynamicDim` for dynamic-input models, plus the gate arguments `minFps`, `maxP95Ms`, `maxModelMb` and `requireProvider`.

Benchmark results report the execution provider the ONNX Runtime session **actually** used. When ORT silently falls back (e.g. CUDA listed as available but its libraries fail to load), `provider_fallback` is `true` and `fallback_reason` carries ORT's own error. `runtime` records the Python interpreter, ONNX Runtime and NumPy versions that produced the numbers.

The Python interpreter is chosen from `DSH_MODEL_DEPLOY_PYTHON`, then `python3`/`python`, then conda/miniforge environments under the home directory — the first that can import `numpy`, `onnx` and `onnxruntime`. DSH's MCP client scrubs inherited `DSH_*` variables, so to pin an interpreter under DSH set it in the patch entry's `env`.

SSH credentials are not passed to the model. Use `~/.ssh/config`, `ssh-agent`, or the operating system's normal SSH credential handling.

## What V0.1 does not do

TensorRT engine building, FP16/INT8 conversion/calibration, Docker/WSL executors, OpenVINO, TFLite, RKNN and NCNN are intentionally deferred. The next major backend target is TensorRT.

## Reliability rule

The project distinguishes:

1. **Static analysis** — facts read from the model
2. **Compatibility observations** — what the detected runtime appears capable of
3. **Measured benchmark** — performance actually measured on the named machine

It does not invent latency numbers for hardware that was not tested.

## License

MIT
