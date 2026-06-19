from __future__ import annotations

import importlib
import sys


def test_factor_eval_package_import_is_lazy_for_scipy_modules() -> None:
    for module_name in (
        "oxq.factor_eval.metrics",
        "oxq.factor_eval.bias",
        "oxq.factor_eval.decay_curve",
    ):
        sys.modules.pop(module_name, None)

    package = importlib.reload(importlib.import_module("oxq.factor_eval"))

    assert package.__all__
    assert "oxq.factor_eval.metrics" not in sys.modules
    assert "oxq.factor_eval.bias" not in sys.modules
    assert "oxq.factor_eval.decay_curve" not in sys.modules


def test_tools_import_does_not_load_factor_eval_optional_modules() -> None:
    for module_name in (
        "oxq.factor_eval.metrics",
        "oxq.factor_eval.bias",
        "oxq.factor_eval.decay_curve",
        "oxq.factor_eval.tearsheet",
    ):
        sys.modules.pop(module_name, None)

    importlib.reload(importlib.import_module("oxq.tools"))

    assert "oxq.factor_eval.metrics" not in sys.modules
    assert "oxq.factor_eval.bias" not in sys.modules
    assert "oxq.factor_eval.decay_curve" not in sys.modules
    assert "oxq.factor_eval.tearsheet" not in sys.modules
