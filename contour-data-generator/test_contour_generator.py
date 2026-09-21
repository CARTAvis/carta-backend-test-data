import importlib.util
import pathlib
import warnings

import numpy as np


path = pathlib.Path(__file__).with_name("contour-generator.py")
spec = importlib.util.spec_from_file_location("contour_generator", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_trace_level_skips_degenerate_edges_without_warnings():
    image = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vertices, indices = module.trace_level(image, 2, 2, 1.0, 0.0, 0.5)

    assert not caught, f"unexpected warnings: {caught}"
    assert vertices == []
    assert indices == []
