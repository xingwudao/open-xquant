"""Spec Validator — P0 validation rules for strategy_spec.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from oxq.spec.schema import StrategySpec


@dataclass
class ValidationResult:
    """Result of validating a strategy spec."""

    status: str  # "pass" | "fail"
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    spec_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "spec_hash": self.spec_hash,
        }


def _err(severity: str, check: str, message: str) -> dict:
    return {"severity": severity, "check": check, "message": message}


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def validate(spec: StrategySpec) -> ValidationResult:
    """Run all P0 validation rules against a StrategySpec.

    Returns a ValidationResult with status 'fail' if any fatal errors exist,
    or 'pass' otherwise (warnings do not cause failure).
    """
    errors: list[dict] = []
    warnings: list[dict] = []

    # --- Research ---
    if not spec.research.hypothesis.strip():
        errors.append(_err("fatal", "hypothesis_missing", "hypothesis is empty — no testable hypothesis"))

    # --- Universe ---
    if not spec.universe.type:
        errors.append(_err("fatal", "universe_missing", "universe.type is missing — cannot define research scope"))
    supported_universe_types = frozenset({"static"})
    if spec.universe.type and spec.universe.type not in supported_universe_types:
        errors.append(
            _err(
                "fatal",
                "universe_type_unsupported",
                f"Universe type '{spec.universe.type}' is not yet supported. "
                f"Only 'static' is available.",
            )
        )
    if spec.universe.type == "static" and not spec.universe.symbols:
        errors.append(_err("fatal", "universe_empty", "static universe has no symbols"))
    if spec.universe.type == "static" and not spec.universe.point_in_time:
        warnings.append(
            _err(
                "warning",
                "static_universe_survivorship",
                "static universe with point_in_time=false may have survivorship bias",
            )
        )

    # --- Data ---
    if not spec.data.price_adjustment:
        errors.append(_err("fatal", "price_adjustment_missing", "data.price_adjustment is missing — price semantics unclear"))

    # --- Signal ---
    if not spec.signal.signal_time:
        errors.append(_err("fatal", "signal_time_missing", "signal.signal_time is missing — cannot detect look-ahead bias"))

    # --- Execution ---
    if not spec.execution.trade_time:
        errors.append(_err("fatal", "trade_time_missing", "execution.trade_time is missing — cannot determine fill timing"))
    if not spec.execution.fill_price_mode:
        errors.append(_err("fatal", "fill_price_mode_missing", "execution.fill_price_mode is missing"))

    # Validate fill mode against known values
    valid_fill_modes = frozenset({"close", "next_open", "mid"})
    if spec.execution.fill_price_mode and spec.execution.fill_price_mode not in valid_fill_modes:
        errors.append(
            _err(
                "fatal",
                "fill_price_mode_invalid",
                f"Unknown fill_price_mode '{spec.execution.fill_price_mode}'. "
                f"Valid: {', '.join(sorted(valid_fill_modes))}",
            )
        )
    expected_trade_time = {
        "close": "close_t",
        "mid": "close_t",
        "next_open": "next_open",
    }.get(spec.execution.fill_price_mode)
    if expected_trade_time and spec.execution.trade_time != expected_trade_time:
        errors.append(
            _err(
                "fatal",
                "execution_timing_mismatch",
                f"execution.trade_time={spec.execution.trade_time} does not match "
                f"fill_price_mode={spec.execution.fill_price_mode}; expected {expected_trade_time}",
            )
        )

    # Fatal: same-bar signal generation and execution
    if spec.signal.signal_time == "close_t":
        if spec.execution.trade_time == "close_t":
            errors.append(
                _err(
                    "fatal",
                    "execution_lag",
                    "signal_time=close_t and trade_time=close_t — "
                    "signal generated and filled on same bar. "
                    "Use trade_time=next_open.",
                )
            )
        if spec.execution.fill_price_mode in ("close", "mid"):
            errors.append(
                _err(
                    "fatal",
                    "execution_lag",
                    f"signal_time=close_t and fill_price_mode={spec.execution.fill_price_mode} — "
                    "signal computed at close but filled at same-bar price "
                    f"({spec.execution.fill_price_mode}). "
                    "Use fill_price_mode=next_open.",
                )
            )

    # --- Cost ---
    if spec.cost.fee_rate == 0.0 and spec.cost.slippage_rate == 0.0:
        errors.append(_err("fatal", "cost_model_missing", "both fee_rate and slippage_rate are zero — zero-cost model is not acceptable"))
    elif spec.cost.fee_rate == 0.0:
        errors.append(_err("fatal", "fee_missing", "fee_rate is zero"))
    elif spec.cost.slippage_rate == 0.0:
        errors.append(_err("fatal", "slippage_missing", "slippage_rate is zero"))

    # --- Validation ---
    if not spec.validation.test_period or len(spec.validation.test_period) < 2:
        errors.append(
            _err("fatal", "oos_missing", "validation.test_period is missing — no out-of-sample validation period")
        )
    if spec.validation.required_oos and (not spec.validation.test_period or not spec.validation.train_period):
        errors.append(
            _err("fatal", "oos_incomplete", "required_oos=true but train_period or test_period missing")
        )
    if spec.validation.test_period and len(spec.validation.test_period) >= 2:
        test_start = _parse_date(spec.validation.test_period[0])
        test_end = _parse_date(spec.validation.test_period[1])
        if test_start is None or test_end is None or test_start > test_end:
            errors.append(
                _err("fatal", "validation_period_order", "validation.test_period must satisfy start <= end")
            )
        if spec.validation.train_period and len(spec.validation.train_period) >= 2:
            train_start = _parse_date(spec.validation.train_period[0])
            train_end = _parse_date(spec.validation.train_period[1])
            if train_start is None or train_end is None or train_start > train_end:
                errors.append(
                    _err("fatal", "validation_period_order", "validation.train_period must satisfy start <= end")
                )
            elif test_start is not None and train_end >= test_start:
                errors.append(
                    _err(
                        "fatal",
                        "validation_period_order",
                        "validation.train_period must end before validation.test_period starts",
                    )
                )

    # --- Benchmark ---
    if not spec.benchmark.symbols:
        warnings.append(_err("warning", "benchmark_missing", "benchmark.symbols is empty — difficult to judge excess return"))

    # --- Parameter count warning ---
    param_count = sum(len(ind.params) for ind in spec.signal.indicators.values())
    if param_count > 10:
        warnings.append(
            _err("warning", "parameter_count", f"signal indicators have {param_count} total params — risk of overfitting")
        )

    # --- Future-data bias: Peak signal ---
    for rule_def in spec.signal.rules.values():
        if rule_def.type == "Peak":
            warnings.append(
                _err(
                    "warning",
                    "peak_future_data",
                    "Peak signal uses shift(-i) which introduces future-data bias. "
                    "Consider using a different signal type for causal backtests.",
                )
            )

    crossover_count = sum(1 for rule_def in spec.signal.rules.values() if rule_def.type == "Crossover")
    if crossover_count > 1:
        errors.append(
            _err(
                "fatal",
                "multiple_crossover_rules",
                "multiple Crossover signal rules are not supported by the spec compiler",
            )
        )

    # Compute hash and determine status
    spec_hash = spec.compute_hash()
    has_fatal = any(e["severity"] == "fatal" for e in errors)

    return ValidationResult(
        status="fail" if has_fatal else "pass",
        errors=errors,
        warnings=warnings,
        spec_hash=spec_hash,
    )
