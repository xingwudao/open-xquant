"""Spec Validator — P0 validation rules for strategy_spec.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field

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

    # Fatal: same-bar signal generation and execution
    if spec.signal.signal_time == "close_t" and spec.execution.trade_time == "close_t":
        errors.append(
            _err(
                "fatal",
                "execution_lag",
                "signal_time=close_t and trade_time=close_t — "
                "signal generated and filled on same bar. "
                "Use trade_time=next_open or next_bar execution.",
            )
        )
    if spec.signal.signal_time == "close_t" and spec.execution.fill_price_mode == "close":
        errors.append(
            _err(
                "fatal",
                "execution_lag",
                "signal_time=close_t and fill_price_mode=close — "
                "signal generated at close and filled at same close price. "
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

    # --- Benchmark ---
    if not spec.benchmark.symbols:
        warnings.append(_err("warning", "benchmark_missing", "benchmark.symbols is empty — difficult to judge excess return"))

    # --- Parameter count warning ---
    param_count = sum(len(ind.params) for ind in spec.signal.indicators.values())
    if param_count > 10:
        warnings.append(
            _err("warning", "parameter_count", f"signal indicators have {param_count} total params — risk of overfitting")
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
