#!/usr/bin/env node
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js'
import { z } from 'zod'

const execFileAsync = promisify(execFile)
const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const python = process.env.DSH_MODEL_DEPLOY_PYTHON || (process.platform === 'win32' ? 'python' : 'python3')

async function runCli(args: string[]) {
  try {
    const { stdout, stderr } = await execFileAsync(
      python,
      ['-m', 'dsh_model_deploy.cli', ...args],
      { cwd: packageRoot, maxBuffer: 16 * 1024 * 1024, timeout: 10 * 60 * 1000 },
    )
    const text = stdout.trim()
    if (!text) throw new Error(stderr.trim() || 'dsh-model-deploy returned no output')
    return JSON.parse(text)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    throw new Error(`dsh-model-deploy execution failed: ${message}. Run dsh-model-deploy-doctor to inspect dependencies.`)
  }
}

const output = (value: unknown) => ({
  content: [{ type: 'text' as const, text: JSON.stringify(value, null, 2) }],
  structuredContent: value as Record<string, unknown>,
})

const server = new McpServer({ name: 'dsh-model-deploy', version: '0.1.0' })

server.registerTool('model_inspect', {
  description: 'Inspect an ONNX model without running inference: shapes, opset, parameter count, operators, size and dynamic inputs.',
  inputSchema: { model: z.string().min(1) },
}, async ({ model }) => output(await runCli(['inspect', model])))

server.registerTool('deployment_environment', {
  description: 'Probe the current machine for OS, CPU, RAM, NVIDIA GPU, ONNX Runtime providers, SSH and SCP.',
}, async () => output(await runCli(['env'])))

server.registerTool('benchmark_local', {
  description: 'Benchmark an ONNX model on the current machine with ONNX Runtime and optionally evaluate deployment constraints.',
  inputSchema: {
    model: z.string().min(1),
    provider: z.string().optional(),
    warmup: z.number().int().min(0).max(1000).default(10),
    runs: z.number().int().min(1).max(10000).default(50),
    minFps: z.number().positive().optional(),
    maxP95Ms: z.number().positive().optional(),
    maxModelMb: z.number().positive().optional(),
  },
}, async ({ model, provider, warmup, runs, minFps, maxP95Ms, maxModelMb }) => {
  const args = ['bench', model, '--warmup', String(warmup), '--runs', String(runs)]
  if (provider) args.push('--provider', provider)
  if (minFps !== undefined) args.push('--min-fps', String(minFps))
  if (maxP95Ms !== undefined) args.push('--max-p95-ms', String(maxP95Ms))
  if (maxModelMb !== undefined) args.push('--max-model-mb', String(maxModelMb))
  return output(await runCli(args))
})

server.registerTool('benchmark_remote_ssh', {
  description: 'Agentlessly benchmark an ONNX model on a POSIX SSH target using its existing Python and ONNX Runtime environment. Credentials remain in the user SSH configuration/agent.',
  inputSchema: {
    target: z.string().min(1).describe('SSH host alias or user@host'),
    model: z.string().min(1),
    provider: z.string().optional(),
    warmup: z.number().int().min(0).max(1000).default(10),
    runs: z.number().int().min(1).max(10000).default(50),
  },
}, async ({ target, model, provider, warmup, runs }) => {
  const args = ['remote-bench', target, model, '--warmup', String(warmup), '--runs', String(runs)]
  if (provider) args.push('--provider', provider)
  return output(await runCli(args))
})

await server.connect(new StdioServerTransport())
