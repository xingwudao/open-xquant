import hashlib
import json
from pathlib import Path

import pytest
import yaml

from oxq.spec.execution import derive_execution_semantics
from oxq.spec.schema import StrategySpec, _dataclass_to_canonical_dict


def test_missing_metrics_section_defaults_to_open_xquant_profile(tmp_path: Path) -> None:
    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: metrics_default
research:
  hypothesis: missing metrics should use the open xquant profile
universe:
  type: static
  symbols: [SPY]
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2021-01-01", "2021-12-31"]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.metrics.profile == "open_xquant_default"
    assert spec.metrics.risk_free_rate == 0.0
    assert spec.metrics.return_type == "simple"
    assert spec.metrics.annualization_days == 252
    assert spec.metrics.calmar_denominator == "max_drawdown"
    assert spec.metrics.evaluation_window == "full"


def test_profile_only_xquant_production_uses_profile_defaults(tmp_path: Path) -> None:
    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: metrics_xquant_defaults
research:
  hypothesis: profile selection should apply profile defaults
universe:
  type: static
  symbols: [SPY]
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  test_period: ["2021-01-01", "2021-12-31"]
metrics:
  profile: xquant_production
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.metrics.profile == "xquant_production"
    assert spec.metrics.risk_free_rate == 0.02
    assert spec.metrics.return_type == "log"
    assert spec.metrics.annualization_days == 252
    assert spec.metrics.calmar_denominator == "max_drawdown"
    assert spec.metrics.evaluation_window == "full"


def test_xquant_metrics_explicit_default_overrides_round_trip(tmp_path: Path) -> None:
    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: metrics_xquant_override
research:
  hypothesis: explicit metrics overrides should survive artifacts
metrics:
  profile: xquant_production
  risk_free_rate: 0.0
  return_type: simple
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)
    out_path = tmp_path / "round_trip.yaml"
    out_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=True), encoding="utf-8")
    reparsed = StrategySpec.from_yaml(out_path)

    assert spec.to_dict()["metrics"]["risk_free_rate"] == 0.0
    assert spec.to_dict()["metrics"]["return_type"] == "simple"
    assert reparsed.metrics.profile == "xquant_production"
    assert reparsed.metrics.risk_free_rate == 0.0
    assert reparsed.metrics.return_type == "simple"


def test_xquant_metrics_profile_only_serializes_effective_defaults(tmp_path: Path) -> None:
    spec = StrategySpec.template(strategy_id="metrics_xquant_profile_only", hypothesis="profile-only metrics serialize defaults")
    spec.metrics.profile = "xquant_production"
    out_path = tmp_path / "round_trip.yaml"

    serialized = spec.to_dict()
    out_path.write_text(yaml.safe_dump(serialized, sort_keys=True), encoding="utf-8")
    reparsed = StrategySpec.from_yaml(out_path)

    assert serialized["metrics"]["risk_free_rate"] == 0.02
    assert serialized["metrics"]["return_type"] == "log"
    assert reparsed.metrics.profile == "xquant_production"
    assert reparsed.metrics.risk_free_rate == 0.02
    assert reparsed.metrics.return_type == "log"
    assert reparsed.compute_hash() == spec.compute_hash()


def test_default_metrics_do_not_change_legacy_spec_hash() -> None:
    spec = StrategySpec.template(strategy_id="legacy_hash", hypothesis="default metrics should preserve legacy hash")
    spec.execution.normalize_lot_size_config()
    legacy_canonical = _dataclass_to_canonical_dict(spec)
    legacy_canonical.pop("metrics", None)
    canonical = json.dumps(legacy_canonical, sort_keys=True, default=str)
    legacy_hash = f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"

    assert spec.compute_hash() == legacy_hash


def test_parse_execution_cash_return_and_lot_size_config(tmp_path: Path) -> None:
    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: execution_fields
research:
  hypothesis: execution fields parse
market:
  calendar: XNYS
universe:
  type: static
  symbols: [SPY]
data:
  provider: local
  price_adjustment: adjusted
  required_columns: [open, high, low, close, volume]
signal:
  signal_time: close_t
portfolio:
  type: EqualWeight
execution:
  trade_time: next_open
  fill_price_mode: next_open
  order_timing: next_session_open
  price_bar: next_session
  price_type: open
  cash_annual_return: 0.025
  lot_size: 1
  lot_size_config:
    default: 100
    by_symbol:
      SPY: 1
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
validation:
  train_period: ["2020-01-01", "2020-12-31"]
  test_period: ["2021-01-01", "2021-12-31"]
  required_oos: true
benchmark:
  symbols: [SPY]
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.execution.order_timing == "next_session_open"
    assert spec.execution.price_bar == "next_session"
    assert spec.execution.price_type == "open"
    assert spec.execution.cash_annual_return == 0.025
    assert spec.execution.lot_size_config.default == 100
    assert spec.execution.lot_size_config.by_symbol == {"SPY": 1}


def test_programmatic_lot_size_round_trip_keeps_hash_stable(tmp_path: Path) -> None:
    spec = StrategySpec.template(strategy_id="lot_size_round_trip", hypothesis="round trip")
    spec.execution.lot_size = 100
    original_hash = spec.compute_hash()

    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=True), encoding="utf-8")

    reparsed = StrategySpec.from_yaml(spec_path)

    assert reparsed.execution.lot_size == 100
    assert reparsed.execution.lot_size_config.default == 100
    assert reparsed.compute_hash() == original_hash


def test_explicit_lot_size_config_default_overrides_legacy_lot_size(tmp_path: Path) -> None:
    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: explicit_lot_config
execution:
  lot_size: 100
  lot_size_config:
    default: 1
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)

    assert spec.execution.lot_size == 100
    assert spec.execution.lot_size_config.default == 1
    assert spec.to_dict()["execution"]["lot_size_config"]["default"] == 1


@pytest.mark.parametrize(
    ("field_name", "message"),
    [
        ("order_timing", "execution.order_timing must be a string"),
        ("price_bar", "execution.price_bar must be a string"),
        ("price_type", "execution.price_type must be a string"),
    ],
)
def test_explicit_execution_fields_must_be_strings(tmp_path: Path, field_name: str, message: str) -> None:
    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(
        f"""
schema_version: "0.1"
strategy_id: invalid_execution_field
execution:
  {field_name}: [invalid]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        StrategySpec.from_yaml(spec_path)


def test_derive_legacy_next_open_execution_semantics() -> None:
    spec = StrategySpec.template(strategy_id="legacy", hypothesis="legacy mapping")
    effective = derive_execution_semantics(spec.execution)

    assert effective.order_timing == "next_session_open"
    assert effective.price_bar == "next_session"
    assert effective.price_type == "open"
    assert effective.fill_price_mode == "next_open"
    assert effective.compatibility_source == "legacy_fill_price_mode"


def test_helper_derives_explicit_execution_semantics_without_legacy_fill_mode() -> None:
    # Validator support for explicit-only specs is added in a later task.
    spec = StrategySpec.template(strategy_id="explicit", hypothesis="explicit mapping")
    spec.execution.order_timing = "next_session_close"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "close"
    spec.execution.fill_price_mode = ""

    effective = derive_execution_semantics(spec.execution)

    assert effective.order_timing == "next_session_close"
    assert effective.price_bar == "next_session"
    assert effective.price_type == "close"
    assert effective.fill_price_mode == "next_close"
    assert effective.compatibility_source == "explicit_fields"


def test_helper_derives_explicit_next_session_hl2_execution_semantics() -> None:
    spec = StrategySpec.template(strategy_id="explicit_hl2", hypothesis="explicit hl2 mapping")
    spec.execution.order_timing = "next_session_hl2"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "hl2"
    spec.execution.fill_price_mode = ""

    effective = derive_execution_semantics(spec.execution)

    assert effective.order_timing == "next_session_hl2"
    assert effective.price_bar == "next_session"
    assert effective.price_type == "hl2"
    assert effective.fill_price_mode == "next_hl2"
    assert effective.compatibility_source == "explicit_fields"


def test_programmatic_explicit_execution_treats_default_fill_mode_as_absent() -> None:
    spec = StrategySpec.template(strategy_id="programmatic_explicit", hypothesis="explicit mapping")
    spec.execution.order_timing = "next_session_close"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "close"

    effective = derive_execution_semantics(spec.execution)

    assert effective.fill_price_mode == "next_close"
    assert effective.compatibility_source == "explicit_fields"


def test_programmatic_explicit_execution_hash_round_trips(tmp_path: Path) -> None:
    spec = StrategySpec.template(strategy_id="programmatic_hash", hypothesis="explicit hash round trip")
    spec.execution.order_timing = "next_session_close"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "close"
    original_hash = spec.compute_hash()
    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=True), encoding="utf-8")

    reparsed = StrategySpec.from_yaml(spec_path)

    assert derive_execution_semantics(reparsed.execution).fill_price_mode == "next_close"
    assert reparsed.compute_hash() == original_hash


def test_yaml_explicit_execution_can_omit_legacy_fill_price_mode(tmp_path: Path) -> None:
    spec_path = tmp_path / "strategy.yaml"
    spec_path.write_text(
        """
schema_version: "0.1"
strategy_id: explicit_yaml
execution:
  trade_time: next_open
  order_timing: next_session_close
  price_bar: next_session
  price_type: close
""",
        encoding="utf-8",
    )

    spec = StrategySpec.from_yaml(spec_path)
    effective = derive_execution_semantics(spec.execution)

    assert spec.execution.fill_price_mode == ""
    assert effective.fill_price_mode == "next_close"
    assert effective.compatibility_source == "explicit_fields"


def test_conflicting_execution_semantics_raise_value_error() -> None:
    spec = StrategySpec.template(strategy_id="conflict", hypothesis="conflict")
    spec.execution.fill_price_mode = "next_open"
    spec.execution._fill_price_mode_explicit = True
    spec.execution.order_timing = "same_session_close"
    spec.execution.price_bar = "same_session"
    spec.execution.price_type = "close"

    with pytest.raises(ValueError, match="execution semantics conflict"):
        derive_execution_semantics(spec.execution)
