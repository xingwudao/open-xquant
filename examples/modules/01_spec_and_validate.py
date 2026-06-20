"""Module example: Spec creation and validation.

Demonstrates creating a strategy_spec.yaml programmatically (SDK)
and validating it against P0 rules. Also shows the equivalent CLI commands.

Run: uv run python examples/modules/01_spec_and_validate.py
"""

from pathlib import Path

from oxq.spec import StrategySpec, validate

OUT_DIR = Path("/tmp/oxq_examples/01_spec")

# ---------------------------------------------------------------------------
# SDK: Create a spec programmatically
# ---------------------------------------------------------------------------

spec = StrategySpec.template(
    strategy_id="sma_crossover",
    hypothesis="Short-term SMA crossing above long-term SMA indicates upward momentum.",
)

# Fill in required fields
spec.name = "SMA Crossover Weekly"
spec.universe.symbols = ["SPY", "QQQ"]
spec.universe.type = "static"
spec.universe.point_in_time = False
spec.data.price_adjustment = "adjusted"

# Add indicators and signals
from oxq.spec.schema import IndicatorDef, SignalRuleDef  # noqa: E402

spec.signal.signal_time = "close_t"
spec.signal.indicators = {
    "sma_fast": IndicatorDef(type="SMA", params={"column": "close", "period": 10}),
    "sma_slow": IndicatorDef(type="SMA", params={"column": "close", "period": 50}),
}
spec.signal.rules = {
    "golden_cross": SignalRuleDef(type="Crossover", params={"fast": "sma_fast", "slow": "sma_slow"}),
}

spec.execution.trade_time = "next_open"
spec.execution.fill_price_mode = "next_open"
spec.execution.order_timing = "next_session_open"
spec.execution.price_bar = "next_session"
spec.execution.price_type = "open"
spec.cost.fee_rate = 0.001
spec.cost.slippage_rate = 0.001
spec.execution.cash_annual_return = 0.0
spec.execution.lot_size_config.default = 1
spec.metrics.profile = "open_xquant_default"
spec.metrics.risk_free_rate = 0.0
spec.metrics.return_type = "simple"
spec.metrics.annualization_days = 252
spec.metrics.calmar_denominator = "max_drawdown"
spec.metrics.evaluation_window = "full"
spec.benchmark.symbols = ["SPY"]
spec.validation.train_period = ["2018-01-01", "2021-12-31"]
spec.validation.test_period = ["2022-01-01", "2025-12-31"]
spec.validation.required_oos = True

# Write spec to file (this is what `oxq spec init` does)
OUT_DIR.mkdir(parents=True, exist_ok=True)
spec_path = OUT_DIR / "strategy_spec.yaml"
import yaml  # noqa: E402

spec_yaml = spec.to_dict()
spec_yaml.setdefault("execution", {}).update(
    {
        "cash_annual_return": spec.execution.cash_annual_return,
        "lot_size_config": {
            "default": spec.execution.lot_size_config.default,
            "by_symbol": dict(spec.execution.lot_size_config.by_symbol),
        },
    }
)
spec_yaml["metrics"] = {
    "profile": spec.metrics.profile,
    "risk_free_rate": spec.metrics.risk_free_rate,
    "return_type": spec.metrics.return_type,
    "annualization_days": spec.metrics.annualization_days,
    "calmar_denominator": spec.metrics.calmar_denominator,
    "evaluation_window": spec.metrics.evaluation_window,
}
spec_path.write_text(
    yaml.dump(spec_yaml, sort_keys=False, allow_unicode=True, default_flow_style=False),
    encoding="utf-8",
)
print(f"[SDK] Spec written to {spec_path}")
print(f"       Hash: {spec.compute_hash()}")

# ---------------------------------------------------------------------------
# SDK: Validate the spec
# ---------------------------------------------------------------------------

result = validate(spec)
print(f"\n[SDK] Validation: {result.status.upper()}")
print(f"       Errors:   {len(result.errors)}")
print(f"       Warnings: {len(result.warnings)}")
for w in result.warnings:
    print(f"         - [{w['check']}] {w['message']}")

# ---------------------------------------------------------------------------
# CLI equivalents (shown as comments — run in terminal)
# ---------------------------------------------------------------------------
print(f"""
{'='*60}
CLI equivalents:
  oxq spec init "SMA crossover weekly" --out {spec_path}
  oxq spec validate {spec_path}
{'='*60}
""")

# ---------------------------------------------------------------------------
# SDK: Deliberately create an invalid spec and see it fail
# ---------------------------------------------------------------------------
bad_spec = StrategySpec.template(strategy_id="bad", hypothesis="")
bad_spec.execution.trade_time = "close_t"
bad_spec.execution.fill_price_mode = "close"
bad_spec.execution.order_timing = "same_session_close"
bad_spec.execution.price_bar = "same_session"
bad_spec.execution.price_type = "close"
bad_spec.signal.signal_time = "close_t"

bad_result = validate(bad_spec)
print(f"[SDK] Invalid spec validation: {bad_result.status.upper()}")
for e in bad_result.errors:
    print(f"       - [{e['check']}] {e['message']}")
