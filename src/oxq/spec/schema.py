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


@dataclass
class SignalSection:
    signal_time: str = "close_t"
    indicators: dict[str, IndicatorDef] = field(default_factory=dict)
    rules: dict[str, SignalRuleDef] = field(default_factory=dict)


@dataclass
class PortfolioSection:
    type: str = "EqualWeight"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RebalanceDef:
    frequency: str = "daily"
    interval_days: int = 1


@dataclass
class ExecutionSection:
    trade_time: str = "next_open"
    fill_price_mode: str = "next_open"
    rebalance: RebalanceDef = field(default_factory=RebalanceDef)
    lot_size: int = 1
    initial_cash: float = 100_000.0


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
    robustness: RobustnessSection = field(default_factory=RobustnessSection)
    decision_policy: DecisionPolicy = field(default_factory=DecisionPolicy)

    def compute_hash(self) -> str:
        """Compute sha256 hash of the spec for reproducibility tracking."""
        from dataclasses import asdict

        canonical = json.dumps(asdict(self), sort_keys=True, default=str)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a nested dict suitable for YAML output."""
        return _dataclass_to_dict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StrategySpec:
        """Load a StrategySpec from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid spec file: {path} — expected YAML dict")
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
        )


# --- Parsing helpers ---


def _parse_research(raw: dict) -> ResearchSection:
    return ResearchSection(
        hypothesis=raw.get("hypothesis", ""),
        rationale=raw.get("rationale", ""),
        author=raw.get("author", ""),
        created_at=raw.get("created_at", ""),
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
        indicators[name] = IndicatorDef(type=defn.get("type", ""), params=defn.get("params", {}))
    rules = {}
    for name, defn in raw.get("rules", {}).items():
        rules[name] = SignalRuleDef(type=defn.get("type", ""), params=defn.get("params", {}))
    return SignalSection(
        signal_time=raw.get("signal_time", "close_t"),
        indicators=indicators,
        rules=rules,
    )


def _parse_portfolio(raw: dict) -> PortfolioSection:
    return PortfolioSection(
        type=raw.get("type", "EqualWeight"),
        params=raw.get("params", {}),
    )


def _parse_execution(raw: dict) -> ExecutionSection:
    rebalance_raw = raw.get("rebalance", {})
    rebalance_frequency = rebalance_raw.get("frequency", "daily")
    return ExecutionSection(
        trade_time=raw.get("trade_time", "next_open"),
        fill_price_mode=raw.get("fill_price_mode", "next_open"),
        rebalance=RebalanceDef(
            frequency=rebalance_frequency,
            interval_days=_parse_int(
                rebalance_raw.get("interval_days", 1),
                "execution.rebalance.interval_days",
            ),
        ),
        lot_size=_parse_int(raw.get("lot_size", 1), "execution.lot_size"),
        initial_cash=_parse_float(raw.get("initial_cash", 100_000.0), "execution.initial_cash"),
    )


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

    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if value is not None:
                if f.default is not MISSING and value == f.default:
                    continue
                if f.default_factory is not MISSING:
                    try:
                        if value == f.default_factory():
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
