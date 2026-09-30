from dsh_model_deploy.gate import evaluate_gate


def test_gate_passes_when_constraints_are_met():
    result = evaluate_gate({"provider": "CUDAExecutionProvider", "metrics": {"fps": 42.0, "p95_ms": 28.0}}, min_fps=30, max_p95_ms=35, max_model_mb=200, model_size_mb=120, required_provider="CUDAExecutionProvider")
    assert result["status"] == "PASS"


def test_gate_fails_when_any_constraint_is_missed():
    result = evaluate_gate({"provider": "CPUExecutionProvider", "metrics": {"fps": 20.0, "p95_ms": 55.0}}, min_fps=30, max_p95_ms=35, required_provider="CUDAExecutionProvider")
    assert result["status"] == "FAIL"
