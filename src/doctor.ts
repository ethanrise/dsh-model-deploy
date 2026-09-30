#!/usr/bin/env node
import { execFileSync } from 'node:child_process'
import { resolvePython } from './python.js'

const python = resolvePython()
console.log(`[info] interpreter: ${python}`)
const checks = [
  ['python', ['--version']],
  ['python modules', ['-c', 'import numpy, onnx, psutil; print("core modules ok")']],
  ['onnxruntime', ['-c', 'import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())']],
]

let failed = false
for (const [name, args] of checks) {
  try {
    const value = execFileSync(python, args as string[], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim()
    console.log(`[ok] ${name}: ${value}`)
  } catch {
    failed = true
    console.error(`[missing] ${name}`)
  }
}
if (failed) {
  console.error('\nInstall Python runtime dependencies with: pip install -e ".[runtime]"')
  process.exitCode = 1
}
