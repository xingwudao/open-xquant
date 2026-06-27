"""Strategy Spec — declarative, versionable, hashable strategy definition."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

_UNSET = object()


def make_strategy_id(description: str, max_length: int = 50) -> str:
    """Create a validator-safe strategy_id from free-form text."""
    slug = description.lower()
    slug = re.sub(r"[^a-z0-9_-]+", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_-")
    return (slug[:max_length].rstrip("_-") or "strategy")


@dataclass
class ResearchSection:
    hypothesis: str = ""
    rationale: str = ""
    author: str = ""
    created_at: str = ""


@dataclass
class MarketSection:
    asset_class: str = "equity"
    region: str = "us"
    currency: str = "USD"
    calendar: str = "XNYS"


@dataclass
class UniverseSection:
    type: str = "static"
    symbols: list[str] = field(default_factory=list)
    point_in_time: bool = False
    survivorship_bias_policy: str = "warn"


@dataclass
class DataSection:
    provider: str = "local"
    data_dir: str = ""
    price_adjustment: str = "adjusted"
    required_columns: list[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    min_start_date: str = ""


@dataclass
class IndicatorDef:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SignalRuleDef:
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    output_domain: list[str] = field(default_factory=list)


@dataclass
class SignalSection:
    signal_time: str = "close_t"
    indicators: dict[str, IndicatorDef] = field(default_factory=dict)
    rules: dict[str, SignalRuleDef] = field(default_factory=dict)


@dataclass
class PortfolioRuleDef:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PortfolioSection:
    type: str = "EqualWeight"
    params: dict[str, Any] = field(default_factory=dict)
    rules: dict[str, PortfolioRuleDef] = field(default_factory=dict)


@dataclass(init=False)
class RebalanceDef:
    frequency: str = "daily"
    interval_days: int = 1
    _interval_days_explicit: bool = field(default=False, repr=False, compare=False, metadata={"serialize": False})

    def __init__(
        self,
        frequency: str = "daily",
        interval_days: Any = _UNSET,
        _interval_days_explicit: bool | None = None,
    ) -> None:
        object.__setattr__(self, "frequency", frequency)
        if interval_days is _UNSET:
            object.__setattr__(self, "interval_days", 1)
            object.__setattr__(self, "_interval_days_explicit", bool(_interval_days_explicit))
            return

        object.__setattr__(self, "interval_days", interval_days)
        if _interval_days_explicit is None:
            object.__setattr__(self, "_interval_days_explicit", True)
        else:
            object.__setattr__(self, "_interval_days_explicit", bool(_interval_days_explicit))

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "interval_days" and hasattr(self, "_interval_days_explicit"):
            object.__setattr__(self, "_interval_days_explicit", True)


@dataclass
class LotSizeConfig:
    default: int | None = None
    by_symbol: dict[str, int] = field(default_factory=dict)


@dataclass
class ExecutionSection:
    trade_time: str = "next_open"
    fill_price_mode: str = "next_open"
    rebalance: RebalanceDef = field(default_factory=RebalanceDef)
    lot_size: int = 1
    order_timing: str = ""
    price_bar: str = ""
    price_type: str = ""
    lot_size_config: LotSizeConfig = field(default_factory=LotSizeConfig)
    cash_annual_return: float = 0.0
    initial_cash: float = 100_000.0
    _fill_price_mode_explicit: bool = field(default=False, repr=False, compare=False, metadata={"serialize": False})

    def normalize_lot_size_config(self) -> None:
        if self.lot_size_config.default is None:
            self.lot_size_config.default = self.lot_size


@dataclass
class CostSection:
    fee_rate: float = 0.0
    fee_min: float = 0.0
    slippage_rate: float = 0.0


@dataclass
class BenchmarkSection:
    symbols: list[str] = field(default_factory=list)


@dataclass
class ValidationSection:
    train_period: list[str] = field(default_factory=list)
    test_period: list[str] = field(default_factory=list)
    required_oos: bool = False


@dataclass(init=False)
class MetricsSection:
    profile: str = "open_xquant_default"
    risk_free_rate: float = 0.0
    return_type: str = "simple"
    annualization_days: int = 252
    calmar_denominator: str = "max_drawdown"
    evaluation_window: str = "full"
    _explicit_fields: set[str] = field(default_factory=set, repr=False, compare=False, metadata={"serialize": False})

    def __init__(
        self,
        profile: str = "open_xquant_default",
        risk_free_rate: Any = _UNSET,
        return_type: Any = _UNSET,
        annualization_days: Any = _UNSET,
        calmar_denominator: Any = _UNSET,
        evaluation_window: Any = _UNSET,
        _explicit_fields: set[str] | None = None,
    ) -> None:
        provided_fields = {"profile"} if profile != "open_xquant_default" else set()
        values = {
            "risk_free_rate": 0.0,
            "return_type": "simple",
            "annualization_days": 252,
            "calmar_denominator": "max_drawdown",
            "evaluation_window": "full",
        }
        for name, value in (
            ("risk_free_rate", risk_free_rate),
            ("return_type", return_type),
            ("annualization_days", annualization_days),
            ("calmar_denominator", calmar_denominator),
            ("evaluation_window", evaluation_window),
        ):
            if value is not _UNSET:
                values[name] = value
                provided_fields.add(name)

        object.__setattr__(self, "profile", profile)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_explicit_fields", set(_explicit_fields) if _explicit_fields is not None else provided_fields)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in {
            "profile",
            "risk_free_rate",
            "return_type",
            "annualization_days",
            "calmar_denominator",
            "evaluation_window",
        } and hasattr(self, "_explicit_fields"):
            self._explicit_fields.add(name)


@dataclass
class RobustnessSection:
    cost_multiplier: list[float] = field(default_factory=list)
    parameter_perturbation: dict[str, list[float | int]] = field(default_factory=dict)
    regime_analysis: bool = False


@dataclass
class DecisionPolicy:
    reject_if: dict[str, Any] = field(default_factory=dict)
    promote_if: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySpec:
    """Declarative strategy specification loaded from strategy_spec.yaml.

    This is the canonical, versionable, hashable representation of a
    trading strategy before it is compiled into executable code.
    """

    schema_version: str = "0.1"
    strategy_id: str = ""
    name: str = ""
    required_oxq_version: str = ""

    research: ResearchSection = field(default_factory=ResearchSection)
    market: MarketSection = field(default_factory=MarketSection)
    universe: UniverseSection = field(default_factory=UniverseSection)
    data: DataSection = field(default_factory=DataSection)
    signal: SignalSection = field(default_factory=SignalSection)
    portfolio: PortfolioSection = field(default_factory=PortfolioSection)
    execution: ExecutionSection = field(default_factory=ExecutionSection)
    cost: CostSection = field(default_factory=CostSection)
    benchmark: BenchmarkSection = field(default_factory=BenchmarkSection)
    validation: ValidationSection = field(default_factory=ValidationSection)
    metrics: MetricsSection = field(default_factory=MetricsSection)
    robustness: RobustnessSection = field(default_factory=RobustnessSection)
    decision_policy: DecisionPolicy = field(default_factory=DecisionPolicy)

    def compute_hash(self) -> str:
        """Compute sha256 hash of the spec for reproducibility tracking."""
        self.execution.normalize_lot_size_config()
        canonical_obj = _dataclass_to_canonical_dict(self)
        if self.metrics == MetricsSection():
            canonical_obj.pop("metrics", None)
        if (
            any((self.execution.order_timing, self.execution.price_bar, self.execution.price_type))
            and self.execution.fill_price_mode == "next_open"
            and not self.execution._fill_price_mode_explicit
        ):
            canonical_obj["execution"]["fill_price_mode"] = ""
        if self.execution.rebalance.interval_days == 1 and self.execution.rebalance._interval_days_explicit:
            canonical_obj.setdefault("execution", {}).setdefault("rebalance", {})["interval_days_explicit"] = True
        canonical = json.dumps(canonical_obj, sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a nested dict suitable for YAML output."""
        self.execution.normalize_lot_size_config()
        return _dataclass_to_dict(self)

    def to_effective_dict(self) -> dict[str, Any]:
        """Serialize the complete effective spec, including parser defaults."""
        self.execution.normalize_lot_size_config()
        return _dataclass_to_canonical_dict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StrategySpec:
        """Load a StrategySpec from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid spec file: {path} — expected YAML dict")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StrategySpec:
        """Load a StrategySpec from a parsed mapping."""
        return cls(
            schema_version=raw.get("schema_version", "0.1"),
            strategy_id=raw.get("strategy_id", ""),
            name=raw.get("name", ""),
            required_oxq_version=raw.get("required_oxq_version", ""),
            research=_parse_research(raw.get("research", {})),
            market=_parse_market(raw.get("market", {})),
            universe=_parse_universe(raw.get("universe", {})),
            data=_parse_data(raw.get("data", {})),
            signal=_parse_signal(raw.get("signal", {})),
            portfolio=_parse_portfolio(raw.get("portfolio", {})),
            execution=_parse_execution(raw.get("execution", {})),
            cost=_parse_cost(raw.get("cost", {})),
            benchmark=_parse_benchmark(raw.get("benchmark", {})),
            validation=_parse_validation(raw.get("validation", {})),
            metrics=_parse_metrics(raw.get("metrics", {})),
            robustness=_parse_robustness(raw.get("robustness", {})),
            decision_policy=_parse_decision_policy(raw.get("decision_policy", {})),
        )

    @classmethod
    def template(cls, strategy_id: str = "", hypothesis: str = "") -> StrategySpec:
        """Create a minimal valid template spec."""
        return cls(
            schema_version="0.1",
            strategy_id=strategy_id,
            name=strategy_id.replace("_", " ").title(),
            research=ResearchSection(hypothesis=hypothesis),
            market=MarketSection(),
            universe=UniverseSection(type="static", symbols=["SPY"]),
            data=DataSection(price_adjustment="adjusted"),
            signal=SignalSection(signal_time="close_t"),
            portfolio=PortfolioSection(type="EqualWeight"),
            execution=ExecutionSection(trade_time="next_open", fill_price_mode="next_open"),
            cost=CostSection(fee_rate=0.001, slippage_rate=0.001),
            benchmark=BenchmarkSection(symbols=["SPY"]),
            validation=ValidationSection(
                train_period=["2018-01-01", "2021-12-31"],
                test_period=["2022-01-01", "2025-12-31"],
                required_oos=True,
            ),
            metrics=MetricsSection(),
        )


# --- Parsing helpers ---


def _parse_research(raw: dict) -> ResearchSection:
    return ResearchSection(
        hypothesis=raw.get("hypothesis", ""),
        rationale=raw.get("rationale", ""),
        author=raw.get("author", ""),
        created_at=_parse_date_string(raw.get("created_at", ""), "research.created_at"),
    )


def _parse_market(raw: dict) -> MarketSection:
    return MarketSection(
        asset_class=raw.get("asset_class", "equity"),
        region=raw.get("region", "us"),
        currency=raw.get("currency", "USD"),
        calendar=raw.get("calendar", "XNYS"),
    )


def _parse_universe(raw: dict) -> UniverseSection:
    return UniverseSection(
        type=raw.get("type", "static"),
        symbols=_parse_str_list(raw.get("symbols", []), "universe.symbols"),
        point_in_time=_parse_bool(raw.get("point_in_time", False), "universe.point_in_time"),
        survivorship_bias_policy=raw.get("survivorship_bias_policy", "warn"),
    )


def _parse_data(raw: dict) -> DataSection:
    return DataSection(
        provider=raw.get("provider", "local"),
        data_dir=raw.get("data_dir", ""),
        price_adjustment=raw.get("price_adjustment", "adjusted"),
        required_columns=_parse_str_list(raw.get("required_columns", ["open", "high", "low", "close", "volume"]), "data.required_columns"),
        min_start_date=raw.get("min_start_date", ""),
    )


def _parse_signal(raw: dict) -> SignalSection:
    indicators = {}
    for name, defn in raw.get("indicators", {}).items():
        indicators[name] = IndicatorDef(
            type=defn.get("type", ""),
            params=_parse_params(defn.get("params", {}), f"signal.indicators.{name}.params"),
        )
    rules = {}
    for name, defn in raw.get("rules", {}).items():
        rules[name] = SignalRuleDef(
            type=defn.get("type", ""),
            params=_parse_params(defn.get("params", {}), f"signal.rules.{name}.params"),
            output_domain=_parse_str_list(defn.get("output_domain", []), f"signal.rules.{name}.output_domain"),
        )
    return SignalSection(
        signal_time=raw.get("signal_time", "close_t"),
        indicators=indicators,
        rules=rules,
    )


def _parse_portfolio(raw: dict) -> PortfolioSection:
    rules = {}
    rules_raw = raw.get("rules", {})
    if rules_raw is None:
        rules_raw = {}
    if not isinstance(rules_raw, dict):
        raise ValueError("portfolio.rules must be a mapping")
    for name, defn in rules_raw.items():
        if not isinstance(defn, dict):
            raise ValueError(f"portfolio.rules.{name} must be a mapping")
        params = dict(_parse_params(defn.get("params", {}), f"portfolio.rules.{name}.params"))
        if str(name) == "rebalance" and "interval_days" in params:
            params["interval_days"] = _parse_int(
                params["interval_days"],
                f"portfolio.rules.{name}.params.interval_days",
            )
        rules[str(name)] = PortfolioRuleDef(
            type=_parse_str(defn.get("type", ""), f"portfolio.rules.{name}.type"),
            params=params,
        )
    return PortfolioSection(
        type=raw.get("type", "EqualWeight"),
        params=_parse_params(raw.get("params", {}), "portfolio.params"),
        rules=rules,
    )


def _parse_execution(raw: dict) -> ExecutionSection:
    rebalance_raw = raw.get("rebalance", {})
    if rebalance_raw is None:
        rebalance_raw = {}
    if not isinstance(rebalance_raw, dict):
        raise ValueError("execution.rebalance must be a mapping")
    rebalance_frequency = rebalance_raw.get("frequency", "daily")
    lot_size = _parse_int(raw.get("lot_size", 1), "execution.lot_size")
    order_timing = _parse_str(raw.get("order_timing", ""), "execution.order_timing")
    price_bar = _parse_str(raw.get("price_bar", ""), "execution.price_bar")
    price_type = _parse_str(raw.get("price_type", ""), "execution.price_type")
    fill_price_mode = _parse_str(
        raw.get("fill_price_mode", "" if any((order_timing, price_bar, price_type)) else "next_open"),
        "execution.fill_price_mode",
    )
    return ExecutionSection(
        trade_time=raw.get("trade_time", "next_open"),
        fill_price_mode=fill_price_mode,
        rebalance=RebalanceDef(
            frequency=rebalance_frequency,
            interval_days=_parse_int(
                rebalance_raw.get("interval_days", 1),
                "execution.rebalance.interval_days",
            ),
            _interval_days_explicit="interval_days" in rebalance_raw,
        ),
        lot_size=lot_size,
        order_timing=order_timing,
        price_bar=price_bar,
        price_type=price_type,
        lot_size_config=_parse_lot_size_config(raw.get("lot_size_config"), lot_size),
        cash_annual_return=_parse_float(raw.get("cash_annual_return", 0.0), "execution.cash_annual_return"),
        initial_cash=_parse_float(raw.get("initial_cash", 100_000.0), "execution.initial_cash"),
        _fill_price_mode_explicit="fill_price_mode" in raw,
    )


def _parse_lot_size_config(raw: object, fallback_lot_size: int) -> LotSizeConfig:
    if raw is None:
        return LotSizeConfig(default=fallback_lot_size)
    if not isinstance(raw, dict):
        raise ValueError("execution.lot_size_config must be a mapping")
    default = _parse_int(raw.get("default", fallback_lot_size), "execution.lot_size_config.default")
    by_symbol_raw = raw.get("by_symbol", {})
    if not isinstance(by_symbol_raw, dict):
        raise ValueError("execution.lot_size_config.by_symbol must be a mapping")
    by_symbol = {
        str(symbol): _parse_int(value, f"execution.lot_size_config.by_symbol.{symbol}")
        for symbol, value in by_symbol_raw.items()
    }
    return LotSizeConfig(default=default, by_symbol=by_symbol)


def _parse_cost(raw: dict) -> CostSection:
    return CostSection(
        fee_rate=_parse_float(raw.get("fee_rate", 0.0), "cost.fee_rate"),
        fee_min=_parse_float(raw.get("fee_min", 0.0), "cost.fee_min"),
        slippage_rate=_parse_float(raw.get("slippage_rate", 0.0), "cost.slippage_rate"),
    )


def _parse_benchmark(raw: dict) -> BenchmarkSection:
    return BenchmarkSection(symbols=_parse_str_list(raw.get("symbols", []), "benchmark.symbols"))


def _parse_validation(raw: dict) -> ValidationSection:
    return ValidationSection(
        train_period=_parse_date_list(raw.get("train_period", []), "validation.train_period"),
        test_period=_parse_date_list(raw.get("test_period", []), "validation.test_period"),
        required_oos=_parse_bool(raw.get("required_oos", False), "validation.required_oos"),
    )


def _parse_metrics(raw: object) -> MetricsSection:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("metrics must be a mapping")
    profile = _parse_str(raw.get("profile", "open_xquant_default"), "metrics.profile")
    profile_defaults = _metrics_profile_defaults(profile)
    return MetricsSection(
        profile=profile,
        risk_free_rate=_parse_float(raw.get("risk_free_rate", profile_defaults["risk_free_rate"]), "metrics.risk_free_rate"),
        return_type=_parse_str(raw.get("return_type", profile_defaults["return_type"]), "metrics.return_type"),
        annualization_days=_parse_int(raw.get("annualization_days", profile_defaults["annualization_days"]), "metrics.annualization_days"),
        calmar_denominator=_parse_str(
            raw.get("calmar_denominator", profile_defaults["calmar_denominator"]),
            "metrics.calmar_denominator",
        ),
        evaluation_window=_parse_str(raw.get("evaluation_window", profile_defaults["evaluation_window"]), "metrics.evaluation_window"),
        _explicit_fields=set(raw.keys()),
    )


def _metrics_profile_defaults(profile: str) -> dict[str, object]:
    if profile == "xquant_production":
        return {
            "risk_free_rate": 0.02,
            "return_type": "log",
            "annualization_days": 252,
            "calmar_denominator": "max_drawdown",
            "evaluation_window": "full",
        }
    return {
        "risk_free_rate": 0.0,
        "return_type": "simple",
        "annualization_days": 252,
        "calmar_denominator": "max_drawdown",
        "evaluation_window": "full",
    }


def _parse_robustness(raw: dict) -> RobustnessSection:
    return RobustnessSection(
        cost_multiplier=raw.get("cost_multiplier", []),
        parameter_perturbation=raw.get("parameter_perturbation", {}),
        regime_analysis=_parse_bool(raw.get("regime_analysis", False), "robustness.regime_analysis"),
    )


def _parse_decision_policy(raw: dict) -> DecisionPolicy:
    reject_if = dict(raw.get("reject_if", {}))
    promote_if = dict(raw.get("promote_if", {}))
    for key in ("oos_sharpe_lt", "max_drawdown_lt"):
        if key in reject_if:
            reject_if[key] = _parse_float(reject_if[key], f"decision_policy.reject_if.{key}")
    for key in ("oos_sharpe_gte", "max_drawdown_gte"):
        if key in promote_if:
            promote_if[key] = _parse_float(promote_if[key], f"decision_policy.promote_if.{key}")
    return DecisionPolicy(
        reject_if=reject_if,
        promote_if=promote_if,
    )


def _parse_str_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return value


def _parse_date_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of dates")
    parsed = []
    for item in value:
        if isinstance(item, datetime):
            parsed.append(item.date().isoformat())
        elif isinstance(item, date):
            parsed.append(item.isoformat())
        elif isinstance(item, str):
            parsed.append(item)
        else:
            raise ValueError(f"{field_name} must be a list of date strings")
    return parsed


def _parse_date_string(value: object, field_name: str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    raise ValueError(f"{field_name} must be a date string")


def _parse_params(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _parse_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _parse_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _parse_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("+", "-")):
            digits = stripped[1:]
        else:
            digits = stripped
        if digits.isdigit():
            return int(stripped)
    raise ValueError(f"{field_name} must be an integer")


def _parse_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dict for serialization."""
    from dataclasses import MISSING, fields, is_dataclass

    if isinstance(obj, MetricsSection) and obj.profile != "open_xquant_default":
        return _effective_metrics_dict(obj)

    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            if f.metadata.get("serialize") is False:
                continue
            value = getattr(obj, f.name)
            if value is not None:
                preserve_explicit_rebalance = (
                    (isinstance(obj, ExecutionSection) and f.name == "rebalance" and getattr(value, "_interval_days_explicit", False))
                    or (isinstance(obj, RebalanceDef) and f.name == "interval_days" and obj._interval_days_explicit)
                )
                preserve_explicit_fill_price_mode = (
                    isinstance(obj, ExecutionSection)
                    and f.name == "fill_price_mode"
                    and obj._fill_price_mode_explicit
                )
                if (
                    f.default is not MISSING
                    and value == f.default
                    and not preserve_explicit_rebalance
                    and not preserve_explicit_fill_price_mode
                ):
                    continue
                if f.default_factory is not MISSING:
                    try:
                        if value == f.default_factory() and not preserve_explicit_rebalance and not preserve_explicit_fill_price_mode:
                            continue
                    except TypeError:
                        pass
                result[f.name] = _dataclass_to_dict(value)
        return result
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def _dataclass_to_canonical_dict(obj: Any) -> Any:
    """Recursively convert dataclass to a complete canonical dict."""
    from dataclasses import fields, is_dataclass

    if isinstance(obj, MetricsSection) and obj.profile != "open_xquant_default":
        return _effective_metrics_dict(obj)

    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            if f.metadata.get("serialize") is False:
                continue
            if isinstance(obj, SignalRuleDef) and f.name == "output_domain" and not obj.output_domain:
                continue
            if isinstance(obj, PortfolioSection) and f.name == "rules" and not obj.rules:
                continue
            result[f.name] = _dataclass_to_canonical_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_canonical_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_canonical_dict(v) for k, v in obj.items()}
    return obj


def _effective_metrics_dict(obj: MetricsSection) -> dict[str, Any]:
    defaults = _metrics_profile_defaults(obj.profile)
    explicit_fields = set(getattr(obj, "_explicit_fields", set()))
    return {
        "profile": obj.profile,
        "risk_free_rate": obj.risk_free_rate if "risk_free_rate" in explicit_fields else defaults["risk_free_rate"],
        "return_type": obj.return_type if "return_type" in explicit_fields else defaults["return_type"],
        "annualization_days": obj.annualization_days
        if "annualization_days" in explicit_fields
        else defaults["annualization_days"],
        "calmar_denominator": obj.calmar_denominator
        if "calmar_denominator" in explicit_fields
        else defaults["calmar_denominator"],
        "evaluation_window": obj.evaluation_window
        if "evaluation_window" in explicit_fields
        else defaults["evaluation_window"],
    }
