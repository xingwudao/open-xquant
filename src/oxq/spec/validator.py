"""Spec Validator — P0 validation rules for strategy_spec.yaml."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

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


def _is_finite_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _unsafe_strategy_id(strategy_id: str) -> bool:
    from pathlib import Path

    return not strategy_id or "/" in strategy_id or "\\" in strategy_id or ".." in Path(strategy_id).parts


def _validate_signal_column_references(spec: StrategySpec) -> list[dict]:
    errors: list[dict] = []
    data_columns = set(spec.data.required_columns)
    indicator_columns = set(spec.signal.indicators.keys())
    seen_signal_columns: set[str] = set()
    available_data_columns = data_columns | indicator_columns

    for rule_name, rule_def in spec.signal.rules.items():
        params = rule_def.params
        if rule_def.type == "Crossover":
            for param_name in ("fast", "slow"):
                column = params.get(param_name)
                if not _is_non_empty_string(column):
                    errors.append(_missing_signal_param(rule_name, param_name))
                elif column not in available_data_columns:
                    errors.append(
                        _err(
                            "fatal",
                            "signal_column_missing",
                            f"signal.rules.{rule_name}.{param_name} references unknown column '{column}'",
                        )
                    )
        elif rule_def.type == "Threshold":
            column = params.get("column")
            if not _is_non_empty_string(column):
                errors.append(_missing_signal_param(rule_name, "column"))
            elif column not in available_data_columns:
                errors.append(
                    _err(
                        "fatal",
                        "signal_column_missing",
                        f"signal.rules.{rule_name}.column references unknown column '{column}'",
                        )
                    )
            errors.extend(_validate_relationship(rule_name, params.get("relationship", "gt")))
        elif rule_def.type == "Comparison":
            for param_name in ("left", "right"):
                column = params.get(param_name)
                if not _is_non_empty_string(column):
                    errors.append(_missing_signal_param(rule_name, param_name))
                elif column not in available_data_columns:
                    errors.append(
                        _err(
                            "fatal",
                            "signal_column_missing",
                            f"signal.rules.{rule_name}.{param_name} references unknown column '{column}'",
                        )
                    )
            errors.extend(_validate_relationship(rule_name, params.get("relationship", "gt")))
        elif rule_def.type == "Formula":
            expr = params.get("expr")
            if not _is_non_empty_string(expr):
                errors.append(_missing_signal_param(rule_name, "expr"))
            else:
                available_formula_columns = available_data_columns | seen_signal_columns
                try:
                    import pandas as pd

                    dummy = pd.DataFrame({column: [1.0] for column in sorted(available_formula_columns)})
                    dummy.eval(expr)
                except Exception as exc:
                    errors.append(
                        _err(
                            "fatal",
                            "signal_column_missing",
                            f"signal.rules.{rule_name}.expr cannot be evaluated: {exc}",
                        )
                    )
        elif rule_def.type == "Composite":
            signals = params.get("signals")
            if (
                not isinstance(signals, list)
                or not signals
                or not all(_is_non_empty_string(signal_name) for signal_name in signals)
            ):
                errors.append(
                    _err(
                        "fatal",
                        "signal_param_missing",
                        f"signal.rules.{rule_name}.signals must be a non-empty list of strings",
                    )
                )
            else:
                for signal_name in signals:
                    if signal_name not in seen_signal_columns:
                        errors.append(
                            _err(
                                "fatal",
                                "signal_column_missing",
                                f"signal.rules.{rule_name}.signals references signal '{signal_name}' "
                                "before it has been computed",
                            )
                        )
            logic = params.get("logic", "and")
            if logic not in {"and", "or"}:
                errors.append(
                    _err(
                        "fatal",
                        "signal_logic_invalid",
                        f"signal.rules.{rule_name}.logic must be 'and' or 'or'",
                    )
                )
        seen_signal_columns.add(rule_name)

    return errors


def _validate_derived_column_names(spec: StrategySpec) -> list[dict]:
    errors: list[dict] = []
    data_columns = set(spec.data.required_columns)
    indicator_names = set(spec.signal.indicators.keys())
    signal_names = set(spec.signal.rules.keys())

    for name in sorted(indicator_names & data_columns):
        errors.append(
            _err(
                "fatal",
                "signal_name_collision",
                f"signal.indicators.{name} must not overwrite raw data column '{name}'",
            )
        )
    for name in sorted(signal_names & data_columns):
        errors.append(
            _err(
                "fatal",
                "signal_name_collision",
                f"signal.rules.{name} must not overwrite raw data column '{name}'",
            )
        )
    for name in sorted(indicator_names & signal_names):
        errors.append(
            _err(
                "fatal",
                "signal_name_collision",
                f"derived column name '{name}' is declared as both indicator and signal",
            )
        )
    return errors


def _validate_optimizer_params(spec: StrategySpec) -> list[dict]:
    errors: list[dict] = []
    params = spec.portfolio.params
    available_columns = set(spec.data.required_columns) | set(spec.signal.indicators.keys())

    if spec.portfolio.type == "TopNRanking":
        errors.extend(_require_optimizer_column("portfolio.params.score_col", params.get("score_col"), available_columns))
        n = params.get("n", 5)
        if not _is_positive_int(n):
            errors.append(_optimizer_param_error("portfolio.params.n must be a positive integer"))
        max_weight = params.get("max_weight", 1.0)
        if not _is_finite_number(max_weight) or not 0.0 < float(max_weight) <= 1.0:
            errors.append(_optimizer_param_error("portfolio.params.max_weight must be in (0, 1]"))
        filter_negative = params.get("filter_negative", True)
        if not isinstance(filter_negative, bool):
            errors.append(_optimizer_param_error("portfolio.params.filter_negative must be boolean"))
    elif spec.portfolio.type == "RiskParity":
        errors.extend(_require_optimizer_column("portfolio.params.volatility_col", params.get("volatility_col"), available_columns))
    elif spec.portfolio.type == "Kelly":
        for param_name in ("win_rate_col", "avg_win_col", "avg_loss_col"):
            errors.extend(_require_optimizer_column(f"portfolio.params.{param_name}", params.get(param_name), available_columns))
        fraction = params.get("fraction", 1.0)
        if not _is_finite_number(fraction) or not 0.0 < float(fraction) <= 1.0:
            errors.append(_optimizer_param_error("portfolio.params.fraction must be in (0, 1]"))
    elif spec.portfolio.type == "PctEquity":
        pct = params.get("pct", 0.10)
        if not _is_finite_number(pct) or not 0.0 < float(pct) <= 1.0:
            errors.append(_optimizer_param_error("portfolio.params.pct must be in (0, 1]"))

    return errors


def _require_optimizer_column(field_name: str, value: object, available_columns: set[str]) -> list[dict]:
    if not _is_non_empty_string(value):
        return [_optimizer_param_error(f"{field_name} is required and must be a non-empty string")]
    if value not in available_columns:
        return [
            _err(
                "fatal",
                "optimizer_column_missing",
                f"{field_name} references unknown column '{value}'",
            )
        ]
    return []


def _optimizer_param_error(message: str) -> dict:
    return _err("fatal", "optimizer_param_invalid", message)


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _missing_signal_param(rule_name: str, param_name: str) -> dict:
    return _err(
        "fatal",
        "signal_param_missing",
        f"signal.rules.{rule_name}.{param_name} is required and must be a non-empty string",
    )


def _validate_relationship(rule_name: str, relationship: object) -> list[dict]:
    if not _is_non_empty_string(relationship):
        return [_err(
            "fatal",
            "signal_relationship_invalid",
            f"signal.rules.{rule_name}.relationship must be a non-empty string",
        )]
    try:
        from oxq.signals._ops import resolve_op

        resolve_op(relationship)
    except ValueError as exc:
        return [_err("fatal", "signal_relationship_invalid", f"signal.rules.{rule_name}.relationship is invalid: {exc}")]
    return []


def _validate_timestamp_rule(rule: object) -> list[dict]:
    if rule in {"month_end", "quarter_end"}:
        return [
            _err(
                "fatal",
                "timestamp_end_rule_unsupported",
                "Timestamp month_end/quarter_end rules depend on future rows and are not supported in audited specs",
            )
        ]
    if rule in {"month_start", "quarter_start"}:
        return []
    if isinstance(rule, str) and rule.startswith("weekday:"):
        try:
            weekday = int(rule.split(":", 1)[1])
        except ValueError:
            weekday = -1
        if 0 <= weekday <= 4:
            return []
    return [
        _err(
            "fatal",
            "timestamp_rule_invalid",
            "Timestamp rule must be month_start, quarter_start, or weekday:N with N in 0..4",
        )
    ]


def _validate_compute_params(spec: StrategySpec) -> list[dict]:
    errors: list[dict] = []
    try:
        from oxq.spec.compiler import _resolve_indicator, _resolve_signal
    except Exception as exc:
        return [_err("fatal", "compute_dry_run_failed", f"cannot initialize compute dry run: {exc}")]

    index = pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC")
    frame = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in spec.data.required_columns}, index=index)
    for indicator_name, indicator_def in spec.signal.indicators.items():
        try:
            indicator = _resolve_indicator(indicator_def.type)()
            frame[indicator_name] = indicator.compute(frame, **indicator_def.params)
        except Exception as exc:
            errors.append(
                _err(
                    "fatal",
                    "compute_dry_run_failed",
                    f"signal.indicators.{indicator_name} cannot compute with declared params: {exc}",
                )
            )

    for signal_name, signal_def in spec.signal.rules.items():
        try:
            signal = _resolve_signal(signal_def.type)()
            frame[signal_name] = signal.compute(frame, **signal_def.params)
        except Exception as exc:
            errors.append(
                _err(
                    "fatal",
                    "compute_dry_run_failed",
                    f"signal.rules.{signal_name} cannot compute with declared params: {exc}",
                )
            )

    return errors


def _unsafe_symbol(symbol: str) -> bool:
    from pathlib import Path

    if not symbol or "/" in symbol or "\\" in symbol:
        return True
    path = Path(symbol)
    return path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)


def validate(spec: StrategySpec) -> ValidationResult:
    """Run all P0 validation rules against a StrategySpec.

    Returns a ValidationResult with status 'fail' if any fatal errors exist,
    or 'pass' otherwise (warnings do not cause failure).
    """
    errors: list[dict] = []
    warnings: list[dict] = []

    if not isinstance(spec.strategy_id, str) or _unsafe_strategy_id(spec.strategy_id):
        errors.append(_err("fatal", "strategy_id_invalid", "strategy_id must be non-empty and must not contain path separators or '..'"))

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
    for symbol in spec.universe.symbols:
        if _unsafe_symbol(symbol):
            errors.append(_err("fatal", "universe_symbol_unsafe", f"universe symbol '{symbol}' is not a safe data symbol"))
    if not isinstance(spec.universe.point_in_time, bool):
        errors.append(_err("fatal", "universe_point_in_time_type", "universe.point_in_time must be boolean"))
    if spec.universe.type == "static" and spec.universe.point_in_time is not True:
        warnings.append(
            _err(
                "warning",
                "static_universe_survivorship",
                "static universe with point_in_time=false may have survivorship bias",
            )
        )

    # --- Market ---
    if spec.market.calendar != "XNYS":
        errors.append(
            _err(
                "fatal",
                "market_calendar_unsupported",
                f"market.calendar={spec.market.calendar} is not supported by the audited local compiler",
            )
        )

    # --- Data ---
    if spec.data.provider != "local":
        errors.append(
            _err(
                "fatal",
                "data_provider_unsupported",
                f"data.provider={spec.data.provider} is not supported by compile_run; use local",
            )
        )
    if not spec.data.price_adjustment:
        errors.append(_err("fatal", "price_adjustment_missing", "data.price_adjustment is missing — price semantics unclear"))
    elif spec.data.price_adjustment != "adjusted":
        errors.append(
            _err(
                "fatal",
                "price_adjustment_unsupported",
                "only data.price_adjustment=adjusted is supported by the local data provider",
            )
        )
    required_columns = set(spec.data.required_columns)
    if "close" not in required_columns:
        errors.append(_err("fatal", "required_columns_missing_close", "data.required_columns must include close"))
    if spec.execution.fill_price_mode in {"next_open", "mid"} and "open" not in required_columns:
        errors.append(
            _err(
                "fatal",
                "required_columns_missing_open",
                f"data.required_columns must include open for fill_price_mode={spec.execution.fill_price_mode}",
            )
        )

    # --- Signal ---
    if not spec.signal.signal_time:
        errors.append(_err("fatal", "signal_time_missing", "signal.signal_time is missing — cannot detect look-ahead bias"))
    elif spec.signal.signal_time != "close_t":
        errors.append(
            _err(
                "fatal",
                "signal_time_unsupported",
                "only signal_time=close_t is supported by the runtime",
            )
        )

    # --- Execution ---
    if not spec.execution.trade_time:
        errors.append(_err("fatal", "trade_time_missing", "execution.trade_time is missing — cannot determine fill timing"))
    if not spec.execution.fill_price_mode:
        errors.append(_err("fatal", "fill_price_mode_missing", "execution.fill_price_mode is missing"))
    if not isinstance(spec.execution.lot_size, int) or isinstance(spec.execution.lot_size, bool) or spec.execution.lot_size <= 0:
        errors.append(_err("fatal", "lot_size_invalid", "execution.lot_size must be a positive integer"))
    if (
        not isinstance(spec.execution.rebalance.interval_days, int)
        or isinstance(spec.execution.rebalance.interval_days, bool)
        or spec.execution.rebalance.interval_days <= 0
    ):
        errors.append(_err("fatal", "rebalance_interval_invalid", "execution.rebalance.interval_days must be a positive integer"))
    if spec.execution.rebalance.frequency != "daily" and spec.execution.rebalance.interval_days <= 1:
        errors.append(
            _err(
                "fatal",
                "rebalance_frequency_unsupported",
                "calendar rebalance frequency is not implemented; set execution.rebalance.interval_days explicitly",
            )
        )
    if not _is_finite_number(spec.execution.initial_cash) or spec.execution.initial_cash <= 0:
        errors.append(_err("fatal", "initial_cash_invalid", "execution.initial_cash must be a positive finite number"))

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
    finite_costs = (
        _is_finite_number(spec.cost.fee_rate)
        and _is_finite_number(spec.cost.slippage_rate)
        and _is_finite_number(spec.cost.fee_min)
    )
    if not finite_costs:
        errors.append(_err("fatal", "cost_model_invalid", "cost fields must be finite numbers"))
    elif spec.cost.fee_rate <= 0.0 and spec.cost.slippage_rate <= 0.0:
        errors.append(_err(
            "fatal",
            "cost_model_missing",
            "fee_rate and slippage_rate must be positive — zero or negative costs are not acceptable",
        ))
    elif spec.cost.fee_rate <= 0.0:
        errors.append(_err("fatal", "fee_missing", "fee_rate must be positive"))
    elif spec.cost.slippage_rate <= 0.0:
        errors.append(_err("fatal", "slippage_missing", "slippage_rate must be positive"))
    if finite_costs and spec.cost.fee_min < 0.0:
        errors.append(_err("fatal", "fee_min_negative", "fee_min must not be negative"))

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
    if spec.data.min_start_date:
        min_start = _parse_date(spec.data.min_start_date)
        requested_start_value = (
            spec.validation.train_period[0]
            if spec.validation.train_period
            else spec.validation.test_period[0]
            if spec.validation.test_period
            else ""
        )
        requested_start = _parse_date(requested_start_value)
        if min_start is None:
            errors.append(_err("fatal", "data_min_start_date", "data.min_start_date must be an ISO date"))
        elif requested_start is not None and min_start > requested_start:
            errors.append(
                _err(
                    "fatal",
                    "data_min_start_date",
                    "data.min_start_date must be earlier than or equal to the first requested validation date",
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

    if spec.signal.rules and spec.portfolio.type not in {"EqualWeight"}:
        errors.append(
            _err(
                "fatal",
                "signal_portfolio_unsupported",
                f"portfolio.type={spec.portfolio.type} with signal.rules is not supported by the spec compiler",
            )
        )

    errors.extend(_validate_derived_column_names(spec))
    errors.extend(_validate_signal_column_references(spec))
    errors.extend(_validate_optimizer_params(spec))

    # --- Future-data bias: Peak signal ---
    for rule_def in spec.signal.rules.values():
        if rule_def.type == "Peak":
            errors.append(
                _err(
                    "fatal",
                    "peak_future_data",
                    "Peak signal uses shift(-i) which introduces future-data bias. "
                    "Consider using a different signal type for causal backtests.",
                )
            )
        if rule_def.type == "Timestamp":
            errors.extend(_validate_timestamp_rule(rule_def.params.get("rule", "month_start")))

    crossover_count = sum(1 for rule_def in spec.signal.rules.values() if rule_def.type == "Crossover")
    if crossover_count > 1:
        errors.append(
            _err(
                "fatal",
                "multiple_crossover_rules",
                "multiple Crossover signal rules are not supported by the spec compiler",
            )
        )

    errors.extend(_validate_compute_params(spec))

    try:
        from oxq.spec.compiler import compile_strategy

        compile_strategy(spec)
    except Exception as exc:
        errors.append(
            _err(
                "fatal",
                "compile_dry_run_failed",
                f"spec cannot be compiled: {exc}",
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
