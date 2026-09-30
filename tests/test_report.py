from dsh_model_deploy.report import render_markdown


def test_comparison_markdown():
    text = render_markdown({"kind": "comparison", "results": [{"model": "/tmp/a.onnx", "provider": "CPUExecutionProvider", "avg_ms": 10, "p95_ms": 12, "fps": 100, "file_size_mb": 3.2}]})
    assert "a.onnx" in text
    assert "100" in text


def test_benchmark_markdown_gate():
    text = render_markdown({"benchmark": {"model": "a.onnx", "target": "local", "provider": "CPUExecutionProvider", "metrics": {"avg_ms": 10, "p95_ms": 12, "fps": 100}}, "gate": {"status": "PASS", "checks": []}})
    assert "Deployment Gate: PASS" in text
