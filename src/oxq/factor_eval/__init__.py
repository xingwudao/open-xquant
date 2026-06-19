"""Factor evaluation public API.

Imports are intentionally lazy so optional dependencies such as scipy and
matplotlib do not break unrelated open-xquant tools.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    # Data layer
    "AlignmentReport": "oxq.factor_eval.bundle",
    "FactorBundle": "oxq.factor_eval.bundle",
    "create_bundle": "oxq.factor_eval.bundle",
    # Preprocessing
    "apply_t1_offset": "oxq.factor_eval.preprocessing",
    "mark_limit_days": "oxq.factor_eval.preprocessing",
    "mark_suspension_days": "oxq.factor_eval.preprocessing",
    # Returns
    "compute_forward_returns": "oxq.factor_eval.returns",
    # Cross-sectional metrics. These require scipy.
    "compute_decay": "oxq.factor_eval.metrics",
    "compute_ic": "oxq.factor_eval.metrics",
    "compute_icir": "oxq.factor_eval.metrics",
    "compute_rank_ic": "oxq.factor_eval.metrics",
    "compute_ts_ic": "oxq.factor_eval.metrics",
    "compute_turnover": "oxq.factor_eval.metrics",
    # Time-series factor metrics.
    "compute_decay_curve": "oxq.factor_eval.decay_curve",
    "compute_hit_rate": "oxq.factor_eval.hit_rate",
    "detect_lookahead_bias": "oxq.factor_eval.bias",
    "compute_profit_loss_ratio": "oxq.factor_eval.profit_loss",
    "compute_cash_period_value": "oxq.factor_eval.cash_period",
    "compute_conditional_analysis": "oxq.factor_eval.conditional",
    "compute_asset_comparison": "oxq.factor_eval.comparison",
    # Output. This requires matplotlib.
    "generate_tearsheet": "oxq.factor_eval.tearsheet",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
