from __future__ import annotations

import json
import platform
import sys

payload = {"python": sys.version.split()[0], "platform": platform.platform(), "modules": {}}
for name in ("numpy", "onnxruntime"):
    try:
        module = __import__(name)
        payload["modules"][name] = getattr(module, "__version__", "installed")
    except Exception as exc:
        payload["modules"][name] = None
        payload.setdefault("errors", {})[name] = str(exc)

try:
    import onnxruntime as ort
    payload["ort_providers"] = ort.get_available_providers()
except Exception:
    payload["ort_providers"] = []

print(json.dumps(payload))
