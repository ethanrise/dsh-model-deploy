import pytest

from dsh_model_deploy.benchmark import _materialize_shape, parse_input_shapes


def test_parse_input_shapes():
    assert parse_input_shapes(["images=1x3x640x640"]) == {"images": [1, 3, 640, 640]}


def test_parse_input_shapes_rejects_invalid():
    with pytest.raises(ValueError):
        parse_input_shapes(["images"])


def test_dynamic_shape_default_and_override():
    assert _materialize_shape(["batch", 3, "height", "width"], None, 2) == [2, 3, 2, 2]
    assert _materialize_shape(["batch", 3, "height", "width"], [1, 3, 640, 640], 1) == [1, 3, 640, 640]
