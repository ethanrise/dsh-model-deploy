#!/usr/bin/env node
import{createRequire}from'module';const require=createRequire(import.meta.url);

// src/doctor.ts
import { execFileSync as execFileSync2 } from "node:child_process";

// src/python.ts
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
var REQUIRED_IMPORT = "import numpy, onnx, onnxruntime";
var CONDA_ROOTS = ["miniforge3", "miniconda3", "anaconda3", "mambaforge", "micromamba"];
function condaPythons() {
  const home = homedir();
  const roots = [];
  for (const name of CONDA_ROOTS) roots.push(path.join(home, name));
  try {
    for (const entry of readdirSync(home, { withFileTypes: true })) {
      if (!entry.isDirectory() || entry.name.startsWith(".")) continue;
      for (const name of CONDA_ROOTS) roots.push(path.join(home, entry.name, name));
    }
  } catch {
  }
  const found = [];
  for (const root of roots) {
    if (!existsSync(root)) continue;
    found.push(path.join(root, "bin", "python"));
    try {
      for (const env of readdirSync(path.join(root, "envs"))) found.push(path.join(root, "envs", env, "bin", "python"));
    } catch {
    }
  }
  return found.filter(existsSync);
}
function candidates() {
  const list = [];
  if (process.env.DSH_MODEL_DEPLOY_PYTHON) list.push(process.env.DSH_MODEL_DEPLOY_PYTHON);
  if (process.platform === "win32") list.push("python", "py");
  else list.push("python3", "python", ...condaPythons());
  return [...new Set(list)];
}
function hasRuntime(python2) {
  try {
    execFileSync(python2, ["-c", REQUIRED_IMPORT], { stdio: "ignore", timeout: 3e4 });
    return true;
  } catch {
    return false;
  }
}
var cached;
function resolvePython() {
  if (cached) return cached;
  const list = candidates();
  cached = list.find(hasRuntime) ?? list[0];
  return cached;
}

// src/doctor.ts
var python = resolvePython();
console.log(`[info] interpreter: ${python}`);
var checks = [
  ["python", ["--version"]],
  ["python modules", ["-c", 'import numpy, onnx, psutil; print("core modules ok")']],
  ["onnxruntime", ["-c", "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"]]
];
var failed = false;
for (const [name, args] of checks) {
  try {
    const value = execFileSync2(python, args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
    console.log(`[ok] ${name}: ${value}`);
  } catch {
    failed = true;
    console.error(`[missing] ${name}`);
  }
}
if (failed) {
  console.error('\nInstall Python runtime dependencies with: pip install -e ".[runtime]"');
  process.exitCode = 1;
}
