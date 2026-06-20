"""ROC timing strategy spec examples.

Shows both fixed-threshold and rolling-quantile ROC timing specs.
"""

from pathlib import Path

import yaml

from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec


def build_fixed_spec() -> StrategySpec:
    spec = StrategySpec.template(
        strategy_id="roc_timing_fixed",
        hypothesis="Buy CSI300 after deep negative ROC and exit after high positive ROC.",
    )
    spec.universe.symbols = ["CSI300"]
    spec.market.calendar = "XSHG"
    spec.signal.indicators = {
        "roc_120": IndicatorDef(type="ROC", params={"column": "close", "period": 120})
    }
    spec.signal.rules = {
        "timing": SignalRuleDef(
            type="ROCTiming",
            params={"column": "roc_120", "mode": "fixed", "bottom": -5.0, "top": 5.0},
        )
    }
    spec.portfolio.type = "SignalToPosition"
    spec.portfolio.params = {"signal": "timing", "buy_weight": 1.0, "sell_weight": 0.0}
    spec.execution.trade_time = "next_open"
    spec.execution.fill_price_mode = "next_open"
    spec.cost.fee_rate = 0.001
    spec.cost.slippage_rate = 0.001
    spec.benchmark.symbols = ["CSI300"]
    spec.validation.train_period = ["2015-01-01", "2022-12-31"]
    spec.validation.test_period = ["2023-01-01", "2026-06-18"]
    return spec


def build_rolling_quantile_spec() -> StrategySpec:
    spec = build_fixed_spec()
    spec.strategy_id = "roc_timing_rolling_quantile"
    spec.signal.rules["timing"].params = {
        "column": "roc_120",
        "mode": "rolling_quantile",
        "q_window": 60,
        "q_bottom": 0.05,
        "q_top": 0.95,
    }
    return spec


if __name__ == "__main__":
    out_dir = Path("/tmp/oxq_examples/roc_timing")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "roc_timing_fixed.yaml").write_text(
        yaml.dump(build_fixed_spec().to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (out_dir / "roc_timing_rolling_quantile.yaml").write_text(
        yaml.dump(build_rolling_quantile_spec().to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(out_dir)
