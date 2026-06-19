from pathlib import Path

import pytest
import yaml

from oxq.spec.execution import derive_execution_semantics
from oxq.spec.schema import StrategySpec


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
    assert effective.fill_price_mode == "close"
    assert effective.compatibility_source == "explicit_fields"


def test_conflicting_execution_semantics_raise_value_error() -> None:
    spec = StrategySpec.template(strategy_id="conflict", hypothesis="conflict")
    spec.execution.fill_price_mode = "next_open"
    spec.execution.order_timing = "same_session_close"
    spec.execution.price_bar = "same_session"
    spec.execution.price_type = "close"

    with pytest.raises(ValueError, match="execution semantics conflict"):
        derive_execution_semantics(spec.execution)
