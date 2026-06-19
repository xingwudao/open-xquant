"""Strategy Spec Validation — pass/fail/warn demo.

Demonstrates spec validation rules by creating 6 specs and validating them.

Run: uv run python examples/strategies/spec_validation_demo.py
"""

from oxq.spec import StrategySpec, validate
from oxq.spec.schema import IndicatorDef, SignalRuleDef


def _print_result(label: str, result):
    print(f"\n{'='*50}")
    print(f"  {label}: {result.status.upper()}")
    for e in result.errors:
        print(f"  FAIL [{e['check']}]: {e['message'][:80]}")
    for w in result.warnings:
        print(f"  WARN [{w['check']}]: {w['message'][:80]}")
    if not result.errors and not result.warnings:
        print("  (no issues)")


# ── 1. Valid SMA crossover ─────────────────────────────────────────────
spec = StrategySpec.template(strategy_id="sma_crossover_valid", hypothesis="Short-term SMA crossover signals upward momentum.")
spec.name = "SMA Crossover (Valid)"
spec.universe.symbols = ["SPY", "QQQ"]
spec.signal.indicators = {
    "sma_fast": IndicatorDef(type="SMA", params={"column": "close", "period": 10}),
    "sma_slow": IndicatorDef(type="SMA", params={"column": "close", "period": 50}),
}
spec.signal.rules = {"golden_cross": SignalRuleDef(type="Crossover", params={"fast": "sma_fast", "slow": "sma_slow"})}
spec.execution.trade_time = "next_open"
spec.execution.fill_price_mode = "next_open"
spec.execution.order_timing = "next_session_open"
spec.execution.price_bar = "next_session"
spec.execution.price_type = "open"
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001
spec.metrics.profile = "open_xquant_default"
spec.benchmark.symbols = ["SPY"]
spec.validation.train_period = ["2018-01-01", "2021-12-31"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
spec.validation.required_oos = True
_print_result("1. sma_crossover_valid (should PASS)", validate(spec))

# ── 2. Unsupported Top-N with signal rules ────────────────────────────
spec = StrategySpec.template(strategy_id="momentum_topn_unsupported", hypothesis="Relative strength has short-term continuation.")
spec.name = "Momentum Top-N (Unsupported)"
spec.universe.symbols = ["SPY", "QQQ", "IWM"]
spec.signal.indicators = {
    "momentum_20": IndicatorDef(type="NdayReturn", params={"column": "close", "period": 20}),
}
spec.signal.rules = {
    "positive_mom": SignalRuleDef(
        type="Threshold",
        params={"column": "momentum_20", "threshold": 0, "relationship": "gt"},
    )
}
spec.portfolio.type = "TopNRanking"
spec.portfolio.params = {"score_col": "momentum_20", "n": 1}
spec.execution.trade_time = "next_open"
spec.execution.fill_price_mode = "next_open"
spec.execution.order_timing = "next_session_open"
spec.execution.price_bar = "next_session"
spec.execution.price_type = "open"
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001
spec.benchmark.symbols = ["SPY"]
spec.validation.train_period = ["2018-01-01", "2021-12-31"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
spec.validation.required_oos = True
_print_result("2. momentum_topn_unsupported (should FAIL)", validate(spec))

# ── 3. Same-bar execution (invalid) ────────────────────────────────────
spec = StrategySpec.template(strategy_id="same_bar_execution_invalid", hypothesis="Deliberately invalid: same-bar execution")
spec.universe.symbols = ["SPY"]
spec.signal.signal_time = "close_t"
spec.execution.trade_time = "close_t"
spec.execution.fill_price_mode = "close"
spec.execution.order_timing = "same_session_close"
spec.execution.price_bar = "same_session"
spec.execution.price_type = "close"
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001
spec.benchmark.symbols = ["SPY"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
_print_result("3. same_bar_execution_invalid (should FAIL)", validate(spec))

# ── 4. Zero cost replay warning ────────────────────────────────────────
spec = StrategySpec.template(strategy_id="zero_cost_replay_warning", hypothesis="Replay-style validation with zero costs.")
spec.universe.symbols = ["SPY"]
spec.signal.signal_time = "close_t"
spec.execution.trade_time = "next_open"
spec.execution.fill_price_mode = "next_open"
spec.execution.order_timing = "next_session_open"
spec.execution.price_bar = "next_session"
spec.execution.price_type = "open"
spec.cost.fee_rate = 0.0
spec.cost.slippage_rate = 0.0
spec.benchmark.symbols = ["SPY"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
_print_result("4. zero_cost_replay_warning (should PASS with warning)", validate(spec))

# ── 5. Static universe warning ─────────────────────────────────────────
spec = StrategySpec.template(strategy_id="static_universe_warning", hypothesis="Static universe may have survivorship bias.")
spec.name = "Static Universe (Warning)"
spec.universe.symbols = ["AAPL", "MSFT", "GOOGL"]
spec.universe.point_in_time = False
spec.signal.indicators = {
    "sma_fast": IndicatorDef(type="SMA", params={"column": "close", "period": 10}),
    "sma_slow": IndicatorDef(type="SMA", params={"column": "close", "period": 50}),
}
spec.signal.rules = {"gc": SignalRuleDef(type="Crossover", params={"fast": "sma_fast", "slow": "sma_slow"})}
spec.execution.trade_time = "next_open"
spec.execution.fill_price_mode = "next_open"
spec.execution.order_timing = "next_session_open"
spec.execution.price_bar = "next_session"
spec.execution.price_type = "open"
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001
spec.benchmark.symbols = ["SPY"]
spec.validation.train_period = ["2018-01-01", "2021-12-31"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
spec.validation.required_oos = True
_print_result("5. static_universe_warning (should PASS with survivorship warning)", validate(spec))

# ── 6. Unsupported calendar (invalid) ─────────────────────────────────
spec = StrategySpec.template(strategy_id="unsupported_calendar_invalid", hypothesis="Deliberately invalid: unsupported calendar.")
spec.market.calendar = "BAD"
spec.universe.symbols = ["SPY"]
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001
spec.benchmark.symbols = ["SPY"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
_print_result("6. unsupported_calendar_invalid (should FAIL)", validate(spec))

print(f"\n{'='*50}")
print("  Done — 6 specs validated")
