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

### Python core

```bash
git clone https://github.com/ethanrise/dsh-model-deploy.git
cd dsh-model-deploy
pip install -e ".[runtime]"
```

For CPU inference, standard `onnxruntime` is enough. GPU execution requires an ONNX Runtime build/provider compatible with the target machine.

### DSH plugin adapter

```bash
npm install
npm run build
npm link
dsh-model-deploy-doctor
```

The MCP adapter invokes the Python core. Set `DSH_MODEL_DEPLOY_PYTHON` if the desired interpreter is not `python3` on Linux/macOS or `python` on Windows.

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
- `benchmark_remote_ssh`

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
