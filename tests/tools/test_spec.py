from __future__ import annotations

import yaml

from oxq.tools.spec import spec_init


def test_spec_init_tool_generates_path_safe_strategy_id(tmp_path) -> None:
    out = tmp_path / "strategy_spec.yaml"

    result = spec_init("SMA/RSI crossover!!!", out=str(out))

    assert result["strategy_id"] == "sma_rsi_crossover"
    spec = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert spec["strategy_id"] == "sma_rsi_crossover"
