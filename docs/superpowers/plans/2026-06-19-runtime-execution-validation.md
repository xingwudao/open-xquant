# Runtime Execution Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PR 1 from the research runtime improvements design: explicit execution semantics, supported market calendars, cash return and lot-size config in specs, and layered validator findings.

**Architecture:** Keep existing specs compatible by adding optional schema fields and deriving effective execution semantics from legacy `fill_price_mode` when new fields are absent. Add a focused execution-semantics helper module used by validator, compiler, and artifact writing. Extend validator findings with dimensions while preserving existing `status`, `errors`, and `warnings` behavior.

**Tech Stack:** Python 3.12+, dataclasses, YAML spec parsing, `exchange_calendars`, pytest, existing open-xquant CLI and compiler.

---

## Baseline

Already verified in the PR 1 worktree:

```bash
uv sync --extra dev
uv run --project "$PWD" python -m pytest tests/spec/test_validator.py tests/spec/test_compiler.py -q
```

Expected baseline:

```text
127 passed
```

## File Structure

Create:

- `src/oxq/spec/execution.py`
  - Effective execution semantics and compatibility mapping.

Modify:

- `src/oxq/spec/schema.py`
  - Add execution fields while keeping legacy fields.
- `src/oxq/spec/validator.py`
  - Add layered dimensions and update calendar/execution/cost validation.
- `src/oxq/spec/compiler.py`
  - Use effective execution settings, cash return, lot-size config, and record execution assumptions.
- `src/oxq/trade/sim_broker.py`
  - Add fill mode support only if needed for new `price_type` values that compiler executes.
- `src/oxq/cli/main.py`
  - Preserve CLI behavior; update output only if validation JSON needs new fields.
- `src/oxq/report/generator.py`
  - Optionally show execution assumptions if artifact exists.

Test:

- `tests/spec/test_execution_semantics.py`
- `tests/spec/test_validator.py`
- `tests/spec/test_compiler.py`
- `tests/trade/test_sim_broker.py`
- `tests/report/test_generator.py`

## Compatibility Rules

Existing specs must keep working unchanged.

Effective execution derivation:

```python
fill_price_mode="next_open" -> order_timing="next_session_open", price_bar="next_session", price_type="open"
fill_price_mode="close" -> order_timing="same_session_close", price_bar="same_session", price_type="close"
fill_price_mode="mid" -> order_timing="same_session_close", price_bar="same_session", price_type="mid"
```

New explicit fields:

```yaml
execution:
  order_timing: next_session_open
  price_bar: next_session
  price_type: open
  cash_annual_return: 0.0
  lot_size_config:
    default: 100
    by_symbol:
      513100.SS: 100
```

Canonical validation:

- If legacy and explicit fields agree, pass.
- If legacy and explicit fields conflict, fail with `execution_semantics_conflict`.
- If explicit fields are present and legacy fields are absent, derive the legacy broker mode only when supported.

## Task 1: Add Execution Semantics Schema

**Files:**

- Create: `src/oxq/spec/execution.py`
- Modify: `src/oxq/spec/schema.py`
- Test: `tests/spec/test_execution_semantics.py`
- Test: `tests/spec/test_validator.py`

- [ ] **Step 1: Write failing tests for parsing new execution fields**

Create `tests/spec/test_execution_semantics.py`:

```python
from pathlib import Path

import pytest

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


def test_derive_legacy_next_open_execution_semantics() -> None:
    spec = StrategySpec.template(strategy_id="legacy", hypothesis="legacy mapping")
    effective = derive_execution_semantics(spec.execution)

    assert effective.order_timing == "next_session_open"
    assert effective.price_bar == "next_session"
    assert effective.price_type == "open"
    assert effective.fill_price_mode == "next_open"
    assert effective.compatibility_source == "legacy_fill_price_mode"


def test_derive_explicit_execution_semantics() -> None:
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
```

- [ ] **Step 2: Run tests to confirm failure**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_execution_semantics.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'oxq.spec.execution'
```

- [ ] **Step 3: Implement execution schema fields**

Modify `src/oxq/spec/schema.py`:

```python
@dataclass
class LotSizeConfig:
    default: int = 1
    by_symbol: dict[str, int] = field(default_factory=dict)


@dataclass
class ExecutionSection:
    trade_time: str = "next_open"
    fill_price_mode: str = "next_open"
    order_timing: str = ""
    price_bar: str = ""
    price_type: str = ""
    rebalance: RebalanceDef = field(default_factory=RebalanceDef)
    lot_size: int = 1
    lot_size_config: LotSizeConfig = field(default_factory=LotSizeConfig)
    initial_cash: float = 100_000.0
    cash_annual_return: float = 0.0
```

Add parser:

```python
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
```

Update `_parse_execution`:

```python
def _parse_execution(raw: dict) -> ExecutionSection:
    rebalance_raw = raw.get("rebalance", {})
    rebalance_frequency = rebalance_raw.get("frequency", "daily")
    lot_size = _parse_int(raw.get("lot_size", 1), "execution.lot_size")
    return ExecutionSection(
        trade_time=raw.get("trade_time", "next_open"),
        fill_price_mode=raw.get("fill_price_mode", "next_open"),
        order_timing=raw.get("order_timing", ""),
        price_bar=raw.get("price_bar", ""),
        price_type=raw.get("price_type", ""),
        rebalance=RebalanceDef(
            frequency=rebalance_frequency,
            interval_days=_parse_int(
                rebalance_raw.get("interval_days", 1),
                "execution.rebalance.interval_days",
            ),
        ),
        lot_size=lot_size,
        lot_size_config=_parse_lot_size_config(raw.get("lot_size_config"), lot_size),
        initial_cash=_parse_float(raw.get("initial_cash", 100_000.0), "execution.initial_cash"),
        cash_annual_return=_parse_float(raw.get("cash_annual_return", 0.0), "execution.cash_annual_return"),
    )
```

- [ ] **Step 4: Create execution semantics helper**

Create `src/oxq/spec/execution.py`:

```python
"""Execution semantics helpers for strategy specs."""

from __future__ import annotations

from dataclasses import dataclass

from oxq.spec.schema import ExecutionSection


@dataclass(frozen=True)
class EffectiveExecution:
    order_timing: str
    price_bar: str
    price_type: str
    fill_price_mode: str
    compatibility_source: str


_LEGACY_MAP: dict[str, EffectiveExecution] = {
    "next_open": EffectiveExecution(
        order_timing="next_session_open",
        price_bar="next_session",
        price_type="open",
        fill_price_mode="next_open",
        compatibility_source="legacy_fill_price_mode",
    ),
    "close": EffectiveExecution(
        order_timing="same_session_close",
        price_bar="same_session",
        price_type="close",
        fill_price_mode="close",
        compatibility_source="legacy_fill_price_mode",
    ),
    "mid": EffectiveExecution(
        order_timing="same_session_close",
        price_bar="same_session",
        price_type="mid",
        fill_price_mode="mid",
        compatibility_source="legacy_fill_price_mode",
    ),
}

_EXPLICIT_TO_FILL_MODE: dict[tuple[str, str, str], str] = {
    ("next_session_open", "next_session", "open"): "next_open",
    ("next_session_close", "next_session", "close"): "close",
    ("next_session_mid", "next_session", "mid"): "mid",
    ("same_session_close", "same_session", "close"): "close",
    ("same_session_close", "same_session", "mid"): "mid",
}


def derive_execution_semantics(execution: ExecutionSection) -> EffectiveExecution:
    explicit_values = (execution.order_timing, execution.price_bar, execution.price_type)
    has_explicit = any(explicit_values)
    has_all_explicit = all(explicit_values)

    if has_explicit and not has_all_explicit:
        raise ValueError("execution semantics conflict: order_timing, price_bar, and price_type must be provided together")

    legacy = _LEGACY_MAP.get(execution.fill_price_mode) if execution.fill_price_mode else None

    if not has_explicit:
        if legacy is None:
            raise ValueError(f"unknown fill_price_mode '{execution.fill_price_mode}'")
        return legacy

    fill_mode = _EXPLICIT_TO_FILL_MODE.get(explicit_values)
    if fill_mode is None:
        raise ValueError(
            "unsupported execution semantics: "
            f"order_timing={execution.order_timing}, price_bar={execution.price_bar}, price_type={execution.price_type}"
        )

    explicit = EffectiveExecution(
        order_timing=execution.order_timing,
        price_bar=execution.price_bar,
        price_type=execution.price_type,
        fill_price_mode=fill_mode,
        compatibility_source="explicit_fields",
    )

    if legacy is not None and (
        legacy.order_timing != explicit.order_timing
        or legacy.price_bar != explicit.price_bar
        or legacy.price_type != explicit.price_type
    ):
        raise ValueError("execution semantics conflict: legacy fill_price_mode disagrees with explicit execution fields")

    return explicit
```

- [ ] **Step 5: Run parsing tests**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_execution_semantics.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add src/oxq/spec/schema.py src/oxq/spec/execution.py tests/spec/test_execution_semantics.py
git commit -m "feat(spec): add explicit execution semantics"
```

## Task 2: Add Layered Validator Findings

**Files:**

- Modify: `src/oxq/spec/validator.py`
- Test: `tests/spec/test_validator.py`
- Test: `tests/spec/test_execution_semantics.py`

- [ ] **Step 1: Write failing validator tests for dimensions**

Append to `tests/spec/test_validator.py`:

```python
def test_validate_findings_include_dimensions_for_static_universe_warning() -> None:
    spec = StrategySpec.template(strategy_id="dimension_warning", hypothesis="dimensions")

    result = validate(spec)

    warning = next(item for item in result.warnings if item["check"] == "static_universe_survivorship")
    assert warning["dimensions"] == ["conservative"]


def test_validate_next_session_mid_is_causal_but_not_conservative() -> None:
    spec = StrategySpec.template(strategy_id="next_mid", hypothesis="next mid is causal")
    spec.execution.fill_price_mode = ""
    spec.execution.trade_time = "next_open"
    spec.execution.order_timing = "next_session_mid"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "mid"

    result = validate(spec)

    assert result.status == "pass"
    assert any(
        item["check"] == "execution_conservatism"
        and item["dimensions"] == ["conservative"]
        for item in result.warnings
    )


def test_validate_same_session_mid_remains_fatal() -> None:
    spec = StrategySpec.template(strategy_id="same_mid", hypothesis="same mid is biased")
    spec.execution.fill_price_mode = "mid"

    result = validate(spec)

    assert result.status == "fail"
    assert any(
        item["check"] == "execution_lag"
        and "causal" in item["dimensions"]
        for item in result.errors
    )


def test_validate_zero_cost_is_warning_when_declared_replay_style() -> None:
    spec = StrategySpec.template(strategy_id="zero_cost_replay", hypothesis="zero cost replay")
    spec.cost.fee_rate = 0.0
    spec.cost.slippage_rate = 0.0
    spec.execution.order_timing = "next_session_open"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "open"

    result = validate(spec)

    assert result.status == "pass"
    assert any(
        item["check"] == "cost_model_zero"
        and set(item["dimensions"]) == {"conservative", "production_consistent"}
        for item in result.warnings
    )
```

- [ ] **Step 2: Run tests to confirm failure**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_validator.py::test_validate_findings_include_dimensions_for_static_universe_warning tests/spec/test_validator.py::test_validate_next_session_mid_is_causal_but_not_conservative tests/spec/test_validator.py::test_validate_same_session_mid_remains_fatal tests/spec/test_validator.py::test_validate_zero_cost_is_warning_when_declared_replay_style -q
```

Expected:

```text
FAILED
```

The first failure should mention missing `dimensions` or invalid current severity.

- [ ] **Step 3: Extend `_err` to carry dimensions**

Modify `src/oxq/spec/validator.py`:

```python
def _err(severity: str, check: str, message: str, dimensions: list[str] | None = None) -> dict:
    return {
        "severity": severity,
        "check": check,
        "message": message,
        "dimensions": dimensions or [],
    }
```

Existing tests that compare check/message should continue passing because the
new key is additive.

- [ ] **Step 4: Add execution validation helper usage**

Import:

```python
from oxq.spec.execution import derive_execution_semantics
```

In the execution section of `validate`, derive effective semantics:

```python
effective_execution = None
try:
    effective_execution = derive_execution_semantics(spec.execution)
except ValueError as exc:
    errors.append(_err("fatal", "execution_semantics_invalid", str(exc), ["executable"]))
```

Use `effective_execution` to validate timing:

```python
if effective_execution is not None and spec.signal.signal_time == "close_t":
    same_session = effective_execution.price_bar == "same_session"
    if same_session:
        errors.append(
            _err(
                "fatal",
                "execution_lag",
                "signal_time=close_t cannot be filled with same-session price",
                ["causal"],
            )
        )
    elif effective_execution.price_type in {"close", "mid", "avg"}:
        warnings.append(
            _err(
                "warning",
                "execution_conservatism",
                f"{effective_execution.price_bar} {effective_execution.price_type} fill is causal but not conservative",
                ["conservative"],
            )
        )
```

Keep legacy `trade_time` mismatch checks only when `fill_price_mode` is used
without explicit fields. Avoid double-reporting both old and new errors.

- [ ] **Step 5: Update static universe and cost findings**

Change static universe warning:

```python
warnings.append(
    _err(
        "warning",
        "static_universe_survivorship",
        "static universe with point_in_time=false may have survivorship bias",
        ["conservative"],
    )
)
```

Change zero-cost logic:

```python
elif spec.cost.fee_rate <= 0.0 and spec.cost.slippage_rate <= 0.0:
    if spec.execution.order_timing or spec.execution.price_bar or spec.execution.price_type:
        warnings.append(_err(
            "warning",
            "cost_model_zero",
            "fee_rate and slippage_rate are zero; acceptable only for explicit replay assumptions",
            ["conservative", "production_consistent"],
        ))
    else:
        errors.append(_err(
            "fatal",
            "cost_model_missing",
            "fee_rate and slippage_rate must be positive — zero or negative costs are not acceptable",
            ["conservative"],
        ))
```

Keep single zero fee or zero slippage as fatal for now, because the replay
case must be explicit and symmetric.

- [ ] **Step 6: Run validator tests**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_validator.py tests/spec/test_execution_semantics.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add src/oxq/spec/validator.py tests/spec/test_validator.py tests/spec/test_execution_semantics.py
git commit -m "feat(spec): classify validator findings by dimension"
```

## Task 3: Support Calendars XNYS, ARCX, XSHG, XSHE

**Files:**

- Modify: `src/oxq/spec/validator.py`
- Modify: `src/oxq/spec/compiler.py`
- Test: `tests/spec/test_validator.py`
- Test: `tests/spec/test_compiler.py`

- [ ] **Step 1: Write failing calendar tests**

Append to `tests/spec/test_validator.py`:

```python
import pytest


@pytest.mark.parametrize("calendar", ["XNYS", "ARCX", "XSHG", "XSHE"])
def test_validate_accepts_supported_market_calendars(calendar: str) -> None:
    spec = StrategySpec.template(strategy_id=f"calendar_{calendar.lower()}", hypothesis="calendar support")
    spec.market.calendar = calendar

    result = validate(spec)

    assert not any(error["check"] == "market_calendar_unsupported" for error in result.errors)


def test_validate_rejects_unknown_market_calendar() -> None:
    spec = StrategySpec.template(strategy_id="bad_calendar", hypothesis="calendar support")
    spec.market.calendar = "BAD"

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "market_calendar_unsupported" for error in result.errors)
```

If an existing `test_validate_rejects_unsupported_market_calendar` exists,
replace it with the second test instead of duplicating it.

- [ ] **Step 2: Run tests to confirm failure**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_validator.py::test_validate_accepts_supported_market_calendars -q
```

Expected:

```text
FAILED
```

At least `ARCX`, `XSHG`, or `XSHE` should be rejected by the current validator.

- [ ] **Step 3: Add supported calendar constant**

In `src/oxq/spec/validator.py` add near the top:

```python
SUPPORTED_MARKET_CALENDARS = frozenset({"XNYS", "ARCX", "XSHG", "XSHE"})
```

Replace the current market calendar check:

```python
if spec.market.calendar not in SUPPORTED_MARKET_CALENDARS:
    errors.append(
        _err(
            "fatal",
            "market_calendar_unsupported",
            f"market.calendar={spec.market.calendar} is not supported by the audited local compiler",
            ["executable"],
        )
    )
```

- [ ] **Step 4: Confirm compiler runtime still resolves calendars**

No compiler code is needed if `_exchange_calendar_sessions` already uses
`exchange_calendars.get_calendar(calendar)`. Add this test to
`tests/spec/test_compiler.py`:

```python
def test_exchange_calendar_sessions_accepts_supported_calendar_names() -> None:
    from oxq.spec.compiler import _exchange_calendar_sessions

    for calendar in ["XNYS", "ARCX", "XSHG", "XSHE"]:
        sessions = _exchange_calendar_sessions(
            pd.Timestamp("2025-01-02"),
            pd.Timestamp("2025-01-10"),
            calendar,
        )
        assert sessions is not None
        assert len(sessions) > 0
```

- [ ] **Step 5: Run calendar tests**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_validator.py::test_validate_accepts_supported_market_calendars tests/spec/test_validator.py::test_validate_rejects_unknown_market_calendar tests/spec/test_compiler.py::test_exchange_calendar_sessions_accepts_supported_calendar_names -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add src/oxq/spec/validator.py tests/spec/test_validator.py tests/spec/test_compiler.py
git commit -m "feat(spec): support core exchange calendars"
```

## Task 4: Apply Cash Return and Lot-Size Config in Compiler

**Files:**

- Modify: `src/oxq/spec/compiler.py`
- Test: `tests/spec/test_compiler.py`

- [ ] **Step 1: Write failing compiler tests for cash return and lot size**

Append to `tests/spec/test_compiler.py`:

```python
def test_compile_run_passes_cash_annual_return_to_engine(monkeypatch, tmp_path: Path) -> None:
    spec = StrategySpec.template(strategy_id="cash_return", hypothesis="cash return")
    spec.execution.cash_annual_return = 0.025

    captured = {}

    class FakeEngine:
        def run(self, *args, **kwargs):
            captured["cash_annual_return"] = kwargs["cash_annual_return"]
            return _minimal_run_result()

    monkeypatch.setattr("oxq.spec.compiler.Engine", FakeEngine)
    monkeypatch.setattr("oxq.spec.compiler._write_artifacts", lambda *args, **kwargs: None)

    compile_run(spec, data_dir=str(tmp_path), out_dir=tmp_path / "runs")

    assert captured["cash_annual_return"] == 0.025


def test_compile_run_uses_lot_size_config_default(monkeypatch, tmp_path: Path) -> None:
    spec = StrategySpec.template(strategy_id="lot_config", hypothesis="lot config")
    spec.execution.lot_size = 1
    spec.execution.lot_size_config.default = 100

    captured = {}

    class FakeEngine:
        def run(self, *args, **kwargs):
            captured["lot_size"] = kwargs["lot_size"]
            return _minimal_run_result()

    monkeypatch.setattr("oxq.spec.compiler.Engine", FakeEngine)
    monkeypatch.setattr("oxq.spec.compiler._write_artifacts", lambda *args, **kwargs: None)

    compile_run(spec, data_dir=str(tmp_path), out_dir=tmp_path / "runs")

    assert captured["lot_size"] == 100
```

If `_minimal_run_result` does not exist in `tests/spec/test_compiler.py`, add:

```python
def _minimal_run_result() -> RunResult:
    portfolio = Portfolio(cash=Decimal("100000"))
    return RunResult(
        strategy_name="minimal",
        portfolio=portfolio,
        trades=[],
        equity_curve=[(pd.Timestamp("2024-01-02", tz="UTC"), 100000.0)],
        signals={},
        indicators={},
        mktdata={},
    )
```

Use the existing `RunResult` and `Portfolio` import style in the file.

- [ ] **Step 2: Run tests to confirm failure**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_compiler.py::test_compile_run_passes_cash_annual_return_to_engine tests/spec/test_compiler.py::test_compile_run_uses_lot_size_config_default -q
```

Expected:

```text
FAILED
```

The cash test should show `cash_annual_return` missing or equal to `0.0`.
The lot test should show `lot_size == 1`.

- [ ] **Step 3: Add effective lot-size helper**

In `src/oxq/spec/compiler.py` add:

```python
def _effective_lot_size(spec: StrategySpec) -> int:
    default = getattr(spec.execution.lot_size_config, "default", None)
    if isinstance(default, int) and not isinstance(default, bool) and default > 0:
        return default
    return spec.execution.lot_size
```

Update `Engine().run` call:

```python
result = engine.run(
    strategy=strategy,
    market=market,
    broker=broker,
    start=start,
    end=end,
    initial_cash=spec.execution.initial_cash,
    lot_size=_effective_lot_size(spec),
    rules=rules,
    data_start=spec.data.min_start_date or None,
    cash_annual_return=spec.execution.cash_annual_return,
)
```

- [ ] **Step 4: Validate lot-size config**

In `src/oxq/spec/validator.py`, add after existing `lot_size` validation:

```python
if (
    not isinstance(spec.execution.lot_size_config.default, int)
    or isinstance(spec.execution.lot_size_config.default, bool)
    or spec.execution.lot_size_config.default <= 0
):
    errors.append(_err("fatal", "lot_size_config_invalid", "execution.lot_size_config.default must be a positive integer", ["executable"]))
for symbol, lot_size in spec.execution.lot_size_config.by_symbol.items():
    if not isinstance(lot_size, int) or isinstance(lot_size, bool) or lot_size <= 0:
        errors.append(
            _err(
                "fatal",
                "lot_size_config_invalid",
                f"execution.lot_size_config.by_symbol.{symbol} must be a positive integer",
                ["executable"],
            )
        )
```

Add validator test:

```python
def test_validate_rejects_invalid_lot_size_config_default() -> None:
    spec = StrategySpec.template(strategy_id="bad_lot_config", hypothesis="lot config")
    spec.execution.lot_size_config.default = 0

    result = validate(spec)

    assert result.status == "fail"
    assert any(error["check"] == "lot_size_config_invalid" for error in result.errors)
```

- [ ] **Step 5: Run compiler and validator tests**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_compiler.py::test_compile_run_passes_cash_annual_return_to_engine tests/spec/test_compiler.py::test_compile_run_uses_lot_size_config_default tests/spec/test_validator.py::test_validate_rejects_invalid_lot_size_config_default -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add src/oxq/spec/compiler.py src/oxq/spec/validator.py tests/spec/test_compiler.py tests/spec/test_validator.py
git commit -m "feat(spec): apply cash return and lot size config"
```

## Task 5: Record Execution Assumptions in Artifacts

**Files:**

- Modify: `src/oxq/spec/compiler.py`
- Test: `tests/spec/test_compiler.py`

- [ ] **Step 1: Write failing artifact test**

Append to `tests/spec/test_compiler.py`:

```python
def test_compile_run_writes_execution_assumptions_artifact(sample_data_dir: Path, tmp_path: Path) -> None:
    spec = StrategySpec.template(strategy_id="execution_artifact", hypothesis="execution assumptions")
    spec.universe.symbols = ["SPY"]
    spec.benchmark.symbols = ["SPY"]
    spec.data.min_start_date = "2024-01-01"
    spec.validation.train_period = ["2024-01-02", "2024-01-05"]
    spec.validation.test_period = ["2024-01-08", "2024-01-10"]
    spec.execution.cash_annual_return = 0.025
    spec.execution.lot_size_config.default = 100

    _, run_dir = compile_run(spec, data_dir=str(sample_data_dir), out_dir=tmp_path / "runs")

    artifact = json.loads((run_dir / "execution_assumptions.json").read_text(encoding="utf-8"))
    assert artifact["calendar"] == "XNYS"
    assert artifact["price_type"] == "open"
    assert artifact["price_bar"] == "next_session"
    assert artifact["order_timing"] == "next_session_open"
    assert artifact["cash_annual_return"] == 0.025
    assert artifact["lot_size_config"]["default"] == 100
```

Use existing sample data fixture names in `tests/spec/test_compiler.py`. If
the fixture has a different name, use the existing fixture that writes `SPY`
parquet data.

- [ ] **Step 2: Run test to confirm failure**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_compiler.py::test_compile_run_writes_execution_assumptions_artifact -q
```

Expected:

```text
FAILED
```

Failure should show `execution_assumptions.json` missing.

- [ ] **Step 3: Write execution artifact**

In `src/oxq/spec/compiler.py`, import:

```python
from oxq.spec.execution import derive_execution_semantics
```

In `_write_artifacts`, after `environment.json`, add:

```python
effective_execution = derive_execution_semantics(spec.execution)
execution_assumptions = {
    "schema_version": 1,
    "calendar": spec.market.calendar,
    "order_timing": effective_execution.order_timing,
    "price_bar": effective_execution.price_bar,
    "price_type": effective_execution.price_type,
    "fill_price_mode": effective_execution.fill_price_mode,
    "compatibility_source": effective_execution.compatibility_source,
    "cash_annual_return": spec.execution.cash_annual_return,
    "lot_size": spec.execution.lot_size,
    "lot_size_config": {
        "default": spec.execution.lot_size_config.default,
        "by_symbol": dict(spec.execution.lot_size_config.by_symbol),
    },
}
(run_dir / "execution_assumptions.json").write_text(
    json.dumps(execution_assumptions, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
```

Add to `artifact_hashes`:

```python
"execution_assumptions.json": _hash_json_file(run_dir / "execution_assumptions.json"),
```

- [ ] **Step 4: Update reproducibility audit expected artifacts if needed**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/audit/test_reproducibility.py tests/spec/test_compiler.py::test_compile_run_writes_execution_assumptions_artifact -q
```

If reproducibility audit has a hard-coded artifact list, add
`execution_assumptions.json` as an optional hashed artifact first. Do not make
old runs without this file fail.

- [ ] **Step 5: Run artifact tests**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_compiler.py::test_compile_run_writes_execution_assumptions_artifact tests/audit/test_reproducibility.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add src/oxq/spec/compiler.py tests/spec/test_compiler.py tests/audit/test_reproducibility.py
git commit -m "feat(spec): record execution assumptions artifact"
```

## Task 6: Wire Effective Execution Into Broker Fill Modes

**Files:**

- Modify: `src/oxq/spec/compiler.py`
- Modify: `src/oxq/trade/sim_broker.py`
- Test: `tests/spec/test_compiler.py`
- Test: `tests/trade/test_sim_broker.py`

- [ ] **Step 1: Write failing test for explicit next-session close**

Append to `tests/spec/test_compiler.py`:

```python
def test_compile_run_uses_explicit_next_session_close_fill_mode(monkeypatch, tmp_path: Path) -> None:
    spec = StrategySpec.template(strategy_id="next_close", hypothesis="explicit next close")
    spec.execution.fill_price_mode = ""
    spec.execution.order_timing = "next_session_close"
    spec.execution.price_bar = "next_session"
    spec.execution.price_type = "close"

    captured = {}

    class FakeBroker:
        def __init__(self, **kwargs):
            captured["fill_price_mode"] = kwargs["fill_price_mode"]

    class FakeEngine:
        def run(self, *args, **kwargs):
            return _minimal_run_result()

    monkeypatch.setattr("oxq.spec.compiler.SimBroker", FakeBroker)
    monkeypatch.setattr("oxq.spec.compiler.Engine", FakeEngine)
    monkeypatch.setattr("oxq.spec.compiler._write_artifacts", lambda *args, **kwargs: None)

    compile_run(spec, data_dir=str(tmp_path), out_dir=tmp_path / "runs")

    assert captured["fill_price_mode"].value == "close"
```

- [ ] **Step 2: Run test to confirm failure**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_compiler.py::test_compile_run_uses_explicit_next_session_close_fill_mode -q
```

Expected:

```text
FAILED
```

Compiler should currently use `spec.execution.fill_price_mode` directly.

- [ ] **Step 3: Use effective execution in compiler**

In `compile_run`, replace:

```python
fill_mode_str = spec.execution.fill_price_mode
```

with:

```python
effective_execution = derive_execution_semantics(spec.execution)
fill_mode_str = effective_execution.fill_price_mode
```

Pass broker calendar only for `next_open` if current broker requires it:

```python
broker = SimBroker(
    fee_model=fee_model,
    slippage_model=slippage_model,
    fill_price_mode=fill_mode,
    market_calendar=spec.market.calendar if fill_mode == FillPriceMode.NEXT_OPEN else None,
)
```

- [ ] **Step 4: Decide whether `next_session_close` is executable in PR 1**

If the existing broker `FillPriceMode.CLOSE` fills same-bar close, do not
silently treat `next_session_close` as executable in `compile_run`.

Implement one of these two choices:

- Preferred for PR 1: validator allows it as causal warning, compiler rejects
  it as not executable until broker supports delayed close fills.
- Alternative: add a new broker mode `NEXT_CLOSE`.

Use the preferred choice unless project maintainers request full delayed close
execution in PR 1.

Preferred implementation:

```python
if effective_execution.order_timing == "next_session_close":
    raise ValueError("next_session_close execution is valid for validation but not yet executable by compile_run")
```

Then change the test expectation to assert `ValueError`.

- [ ] **Step 5: Run compiler tests**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_compiler.py::test_compile_run_uses_explicit_next_session_close_fill_mode tests/spec/test_validator.py::test_validate_next_session_mid_is_causal_but_not_conservative -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add src/oxq/spec/compiler.py src/oxq/trade/sim_broker.py tests/spec/test_compiler.py tests/trade/test_sim_broker.py
git commit -m "feat(spec): compile effective execution semantics"
```

## Task 7: Report Execution Assumptions

**Files:**

- Modify: `src/oxq/report/generator.py`
- Test: `tests/report/test_generator.py`

- [ ] **Step 1: Write failing report test**

Append to `tests/report/test_generator.py`:

```python
def test_report_includes_execution_assumptions_when_artifact_exists(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        """
schema_version: "0.1"
strategy_id: report_execution
research:
  hypothesis: report execution assumptions
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
  cash_annual_return: 0.025
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
benchmark:
  symbols: [SPY]
validation:
  train_period: ["2020-01-01", "2020-12-31"]
  test_period: ["2021-01-01", "2021-12-31"]
""",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        '{"run_id": "run", "total_return": 0.0, "annualized_return": 0.0, "annualized_volatility": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0, "trade_count": 0, "cost_paid": 0.0}',
        encoding="utf-8",
    )
    (run_dir / "execution_assumptions.json").write_text(
        '{"order_timing": "next_session_open", "price_bar": "next_session", "price_type": "open", "cash_annual_return": 0.025, "lot_size_config": {"default": 100, "by_symbol": {}}}',
        encoding="utf-8",
    )
    _write_minimal_audit_files(run_dir)

    report = generate_report(run_dir)

    assert "Execution Assumptions" in report
    assert "next_session_open" in report
    assert "cash annual return" in report.lower()
```

If `tests/report/test_generator.py` already has helper functions for minimal
run directories, use the existing helper instead of adding `_write_minimal_audit_files`.

- [ ] **Step 2: Run test to confirm failure**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/report/test_generator.py::test_report_includes_execution_assumptions_when_artifact_exists -q
```

Expected:

```text
FAILED
```

The report should not yet contain the new section.

- [ ] **Step 3: Add report section**

In `src/oxq/report/generator.py`, read optional artifact:

```python
execution_assumptions = {}
execution_assumptions_path = run_path / "execution_assumptions.json"
if execution_assumptions_path.exists():
    execution_assumptions = json.loads(execution_assumptions_path.read_text(encoding="utf-8"))
```

After "Data and Execution Assumptions", add:

```python
if execution_assumptions:
    lines.append("## Execution Assumptions")
    lines.append("")
    lines.append(f"- **Order Timing**: {execution_assumptions.get('order_timing', 'N/A')}")
    lines.append(f"- **Price Bar**: {execution_assumptions.get('price_bar', 'N/A')}")
    lines.append(f"- **Price Type**: {execution_assumptions.get('price_type', 'N/A')}")
    lines.append(f"- **Cash Annual Return**: {_format_percent(execution_assumptions.get('cash_annual_return'))}")
    lot_config = execution_assumptions.get("lot_size_config", {})
    if isinstance(lot_config, dict):
        lines.append(f"- **Default Lot Size**: {lot_config.get('default', 'N/A')}")
    lines.append("")
```

Use a numbered heading if the report generator's current section numbering
must remain consistent. Keep old reports without artifact working.

- [ ] **Step 4: Run report tests**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/report/test_generator.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 5: Commit Task 7**

Run:

```bash
git add src/oxq/report/generator.py tests/report/test_generator.py
git commit -m "feat(report): include execution assumptions"
```

## Task 8: Full PR Verification

**Files:**

- All changed files.

- [ ] **Step 1: Run focused test suite**

Run:

```bash
uv run --project "$PWD" python -m pytest tests/spec/test_execution_semantics.py tests/spec/test_validator.py tests/spec/test_compiler.py tests/trade/test_sim_broker.py tests/report/test_generator.py tests/audit/test_reproducibility.py -q
```

Expected:

```text
all tests passed
```

- [ ] **Step 2: Run lint**

Run:

```bash
uv run --project "$PWD" ruff check src tests
```

Expected:

```text
All checks passed!
```

- [ ] **Step 3: Run CLI smoke validation**

Create `/tmp/oxq-runtime-validation-spec.yaml`:

```yaml
schema_version: "0.1"
strategy_id: runtime_validation_smoke
name: Runtime Validation Smoke
research:
  hypothesis: Explicit execution semantics validate.
market:
  asset_class: equity
  region: us
  currency: USD
  calendar: XNYS
universe:
  type: static
  symbols: [SPY]
  point_in_time: false
data:
  provider: local
  price_adjustment: adjusted
  required_columns: [open, high, low, close, volume]
signal:
  signal_time: close_t
  indicators: {}
  rules: {}
portfolio:
  type: EqualWeight
execution:
  fill_price_mode: next_open
  order_timing: next_session_open
  price_bar: next_session
  price_type: open
  lot_size: 1
  lot_size_config:
    default: 1
  initial_cash: 100000
  cash_annual_return: 0.0
cost:
  fee_rate: 0.001
  slippage_rate: 0.001
benchmark:
  symbols: [SPY]
validation:
  train_period: ["2020-01-01", "2020-12-31"]
  test_period: ["2021-01-01", "2021-12-31"]
  required_oos: true
```

Run:

```bash
uv run --project "$PWD" oxq spec validate /tmp/oxq-runtime-validation-spec.yaml --json
```

Expected:

```text
"status": "pass"
```

- [ ] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected:

```text
clean
```

or only intentional untracked scratch files outside the repository.

## Self-Review Checklist

- [ ] Existing specs still parse and validate.
- [ ] Existing `fill_price_mode` behavior remains backward compatible.
- [ ] New execution fields are optional.
- [ ] Unsupported explicit execution is clear in validation or compiler error.
- [ ] Validator findings preserve existing top-level status.
- [ ] Findings include dimensions without breaking old consumers.
- [ ] `cash_annual_return` reaches `Engine.run`.
- [ ] `lot_size_config.default` reaches `Engine.run`.
- [ ] Supported calendars are allowed in validator.
- [ ] Execution assumptions are written as a reproducible artifact.
- [ ] Report remains usable for old runs without the new artifact.
