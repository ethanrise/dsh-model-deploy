#!/usr/bin/env node
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { z } from 'zod'
import { resolvePython } from './python.js'

const execFileAsync = promisify(execFile)
const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

/** Last non-empty line of stderr: the Python exception message, without the traceback. */
function lastLine(text: string) {
  return text.trim().split('\n').map(line => line.trim()).filter(Boolean).pop() ?? ''
}

async function runCli(args: string[]) {
  const python = resolvePython()
  let stdout = ''
  let stderr = ''
  try {
    ({ stdout, stderr } = await execFileAsync(
      python,
      ['-m', 'dsh_model_deploy.cli', ...args],
      { cwd: packageRoot, maxBuffer: 16 * 1024 * 1024, timeout: 10 * 60 * 1000 },
    ))
  } catch (error) {
    const failure = error as { stdout?: string, stderr?: string, killed?: boolean, code?: string | number }
    // The CLI reports failures as one JSON line on stdout: {"error", "error_type"}.
    try {
      const payload = JSON.parse(lastLine(failure.stdout ?? ''))
      if (payload?.error) throw new Error(`${payload.error_type}: ${payload.error}`)
    } catch (parsed) {
      if (parsed instanceof Error && !(parsed instanceof SyntaxError)) throw parsed
    }
    if (failure.killed) throw new Error('benchmark timed out after 10 minutes')
    if (failure.code === 'ENOENT') throw new Error(`Python interpreter not found: ${python}. Run dsh-model-deploy-doctor.`)
    const detail = lastLine(failure.stderr ?? '') || (error instanceof Error ? error.message : String(error))
    throw new Error(`${detail} (interpreter: ${python}; run dsh-model-deploy-doctor if dependencies are missing)`)
  }
  const text = stdout.trim()
  if (!text) throw new Error(lastLine(stderr) || 'dsh-model-deploy returned no output')
  return JSON.parse(text)
}

const output = (value: unknown) => ({
  content: [{ type: 'text' as const, text: JSON.stringify(value, null, 2) }],
  structuredContent: value as Record<string, unknown>,
})

// The server runs with the plugin directory as cwd and cannot see the session's
// workspace, so a relative path would silently resolve against the wrong directory.
const modelPath = z.string().min(1)
  .refine(value => path.isAbsolute(value) || value.startsWith('~/'), {
    message: 'model must be an absolute path (or start with ~/); relative paths would resolve against the plugin install directory, not your workspace',
  })
  .describe('Absolute path to the .onnx model file')

const gateInputs = {
  minFps: z.number().positive().optional(),
  maxP95Ms: z.number().positive().optional(),
  maxModelMb: z.number().positive().optional(),
  requireProvider: z.string().optional()
    .describe('Gate check: fail unless this execution provider was actually used (e.g. CUDAExecutionProvider)'),
}

function gateArgs({ minFps, maxP95Ms, maxModelMb, requireProvider }: { minFps?: number, maxP95Ms?: number, maxModelMb?: number, requireProvider?: string }) {
  const args: string[] = []
  if (minFps !== undefined) args.push('--min-fps', String(minFps))
  if (maxP95Ms !== undefined) args.push('--max-p95-ms', String(maxP95Ms))
  if (maxModelMb !== undefined) args.push('--max-model-mb', String(maxModelMb))
  if (requireProvider) args.push('--require-provider', requireProvider)
  return args
}

const shapeInputs = {
  inputShapes: z.record(z.string(), z.array(z.number().int().positive()).min(1)).optional()
    .describe('Per-input shape overrides for dynamic models, e.g. {"images": [1, 3, 640, 640]}'),
  defaultDynamicDim: z.number().int().positive().optional()
    .describe('Value substituted for dynamic dimensions without an explicit override (default 1)'),
}

function shapeArgs(inputShapes?: Record<string, number[]>, defaultDynamicDim?: number) {
  const args: string[] = []
  for (const [name, shape] of Object.entries(inputShapes ?? {})) args.push('--input-shape', `${name}=${shape.join('x')}`)
  if (defaultDynamicDim !== undefined) args.push('--default-dynamic-dim', String(defaultDynamicDim))
  return args
}

export function createServer() {
  const server = new McpServer({ name: 'dsh-model-deploy', version: '0.1.0' })

  const readOnlyLocal = {
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: false,
  }

  const readOnlyRemote = {
    readOnlyHint: true,
    destructiveHint: false,
    idempotentHint: true,
    openWorldHint: true,
  }

  const remoteBenchmark = {
    // The runner/model are copied to a temporary directory and executed remotely.
    readOnlyHint: false,
    destructiveHint: false,
    idempotentHint: false,
    openWorldHint: true,
  }

  server.registerTool('model_inspect', {
    description: 'Inspect an ONNX model without running inference: shapes, opset, parameter count, operators, size and dynamic inputs.',
    inputSchema: { model: modelPath },
    annotations: readOnlyLocal,
  }, async ({ model }) => output(await runCli(['inspect', model])))

  server.registerTool('deployment_environment', {
    description: 'Probe the current machine for OS, CPU, RAM, NVIDIA GPU, ONNX Runtime providers, SSH and SCP.',
    inputSchema: {},
    annotations: readOnlyLocal,
  }, async () => output(await runCli(['env'])))

  server.registerTool('benchmark_local', {
    description: 'Benchmark an ONNX model on the current machine with ONNX Runtime and optionally evaluate deployment constraints. Without `provider`, CUDA is tried when listed and CPU is used if it fails to load.',
    inputSchema: {
      model: modelPath,
      provider: z.string().optional(),
      warmup: z.number().int().min(0).max(1000).default(10),
      runs: z.number().int().min(1).max(10000).default(50),
      ...gateInputs,
      ...shapeInputs,
    },
    annotations: readOnlyLocal,
  }, async ({ model, provider, warmup, runs, inputShapes, defaultDynamicDim, ...gate }) => {
    const args = ['bench', model, '--warmup', String(warmup), '--runs', String(runs), ...shapeArgs(inputShapes, defaultDynamicDim), ...gateArgs(gate)]
    if (provider) args.push('--provider', provider)
    return output(await runCli(args))
  })

  server.registerTool('ssh_preflight', {
    description: 'Check whether an SSH target is benchmark-ready (python3, numpy, onnxruntime and its providers) without copying anything. Run before benchmark_remote_ssh.',
    inputSchema: { target: z.string().min(1).describe('SSH host alias or user@host') },
    annotations: readOnlyRemote,
  }, async ({ target }) => output(await runCli(['ssh-preflight', target])))

  server.registerTool('benchmark_remote_ssh', {
    description: 'Agentlessly benchmark an ONNX model on a POSIX SSH target using its existing Python and ONNX Runtime environment. Credentials remain in the user SSH configuration/agent.',
    inputSchema: {
      target: z.string().min(1).describe('SSH host alias or user@host'),
      model: modelPath,
      provider: z.string().optional(),
      warmup: z.number().int().min(0).max(1000).default(10),
      runs: z.number().int().min(1).max(10000).default(50),
      ...gateInputs,
      ...shapeInputs,
    },
    annotations: remoteBenchmark,
  }, async ({ target, model, provider, warmup, runs, inputShapes, defaultDynamicDim, ...gate }) => {
    const args = ['remote-bench', target, model, '--warmup', String(warmup), '--runs', String(runs), ...shapeArgs(inputShapes, defaultDynamicDim), ...gateArgs(gate)]
    if (provider) args.push('--provider', provider)
    return output(await runCli(args))
  })

  return server
}

const entrypoint = process.argv[1]
if (entrypoint && path.resolve(entrypoint) === fileURLToPath(import.meta.url)) {
  await createServer().connect(new StdioServerTransport())
}
