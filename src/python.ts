import { execFileSync } from 'node:child_process'
import { existsSync, readdirSync } from 'node:fs'
import { homedir } from 'node:os'
import path from 'node:path'

// DSH's mcp-client scrubs DSH_* variables from the inherited environment, so an
// exported DSH_MODEL_DEPLOY_PYTHON may never reach this process. When the
// variable is absent (or unusable), probe likely interpreters and pick the first
// one that can import the runtime dependencies.
const REQUIRED_IMPORT = 'import numpy, onnx, onnxruntime'
const CONDA_ROOTS = ['miniforge3', 'miniconda3', 'anaconda3', 'mambaforge', 'micromamba']

function condaPythons(): string[] {
  const home = homedir()
  // ~/<root> and one level deeper (e.g. ~/app/miniforge3)
  const roots: string[] = []
  for (const name of CONDA_ROOTS) roots.push(path.join(home, name))
  try {
    for (const entry of readdirSync(home, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name.startsWith('.')) continue
      for (const name of CONDA_ROOTS) roots.push(path.join(home, entry.name, name))
    }
  } catch {
    // unreadable home: skip conda discovery
  }
  const found: string[] = []
  for (const root of roots) {
    if (!existsSync(root)) continue
    found.push(path.join(root, 'bin', 'python'))
    try {
      for (const env of readdirSync(path.join(root, 'envs'))) found.push(path.join(root, 'envs', env, 'bin', 'python'))
    } catch {
      // no envs directory
    }
  }
  return found.filter(existsSync)
}

function candidates(): string[] {
  const list: string[] = []
  if (process.env.DSH_MODEL_DEPLOY_PYTHON) list.push(process.env.DSH_MODEL_DEPLOY_PYTHON)
  if (process.platform === 'win32') list.push('python', 'py')
  else list.push('python3', 'python', ...condaPythons())
  return [...new Set(list)]
}

function hasRuntime(python: string): boolean {
  try {
    execFileSync(python, ['-c', REQUIRED_IMPORT], { stdio: 'ignore', timeout: 30_000 })
    return true
  } catch {
    return false
  }
}

let cached: string | undefined

/** Interpreter used to run the Python core; falls back to the platform default when none qualifies. */
export function resolvePython(): string {
  if (cached) return cached
  const list = candidates()
  cached = list.find(hasRuntime) ?? list[0]
  return cached
}
