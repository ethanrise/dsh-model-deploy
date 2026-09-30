from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import onnx


def _shape(value_info: Any) -> list[int | str | None]:
    dims: list[int | str | None] = []
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return dims
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param") and dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return dims


def inspect_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path).expanduser().resolve()
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    initializer_names = {item.name for item in model.graph.initializer}
    parameter_count = 0
    for tensor in model.graph.initializer:
        count = 1
        for dim in tensor.dims:
            count *= int(dim)
        parameter_count += count

    inputs = [
        {"name": item.name, "shape": _shape(item), "elem_type": item.type.tensor_type.elem_type}
        for item in model.graph.input
        if item.name not in initializer_names
    ]
    outputs = [
        {"name": item.name, "shape": _shape(item), "elem_type": item.type.tensor_type.elem_type}
        for item in model.graph.output
    ]
    operators = Counter(node.op_type for node in model.graph.node)

    return {
        "path": str(model_path),
        "file_size_mb": round(model_path.stat().st_size / (1024 ** 2), 3),
        "ir_version": model.ir_version,
        "opset": max((op.version for op in model.opset_import), default=None),
        "parameter_count": parameter_count,
        "node_count": len(model.graph.node),
        "inputs": inputs,
        "outputs": outputs,
        "operators": dict(operators.most_common()),
        "dynamic_inputs": any(
            any(not isinstance(dim, int) for dim in item["shape"]) for item in inputs
        ),
    }
