"""B1 spec end to end: oversold FormulaEvent entry, FormulaExitRule trend exit."""

from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from oxq.spec.compiler import compile_run
from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate

SPEC_PATH = Path(__file__).resolve().parents[2] / "examples" / "strategies" / "zettaranc_b1_spec.yaml"


def _xshg_sessions(n: int) -> pd.DatetimeIndex:
    sessions = xcals.get_calendar("XSHG").sessions_in_range("2021-01-04", "2024-12-31")
    return pd.DatetimeIndex(sessions[:n]).tz_localize("UTC")


def _shaped_bars() -> pd.DataFrame:
    # 220 bars +0.4%/day, 5 bars -2.5%/day closing on the low with half volume
    # (J reaches about -23 on the third dip bar), then 160 bars -0.5%/day so the
    # white line falls below the yellow line.
    up = [10.0 * 1.004**i for i in range(220)]
    dip = [up[-1] * 0.975 ** (i + 1) for i in range(5)]
    down = [dip[-1] * 0.995 ** (i + 1) for i in range(160)]
    close = np.array(up + dip + down)
    n = len(close)
    high = close * 1.005
    low = close * 0.995
    low[220:225] = close[220:225]
    volume = np.full(n, 2e6)
    volume[220:225] = 1e6
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=_xshg_sessions(n),
    )


def test_example_spec_validates() -> None:
    result = validate(StrategySpec.from_yaml(SPEC_PATH))
    assert result.status != "fail", result.errors


def test_b1_spec_buys_on_oversold_and_exits_on_trend_break(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bars = _shaped_bars()
    bars.to_parquet(data_dir / "AAA.parquet")

    spec = StrategySpec.from_yaml(SPEC_PATH)
    spec.universe.symbols = ["AAA"]
    spec.data.min_start_date = str(bars.index[0].date())  # load warm-up bars before the test window
    spec.validation.test_period = [str(bars.index[130].date()), str(bars.index[-1].date())]
    spec.portfolio.rules.pop("no_rise_3d")  # isolate the trend exit path
    spec.portfolio.rules.pop("one_bar_stop")

    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")
    orders = pd.read_csv(run_dir / "orders.csv")
    sides = [str(s).lower() for s in orders["side"]]
    assert "buy" in sides
    first_buy = sides.index("buy")
    assert "sell" in sides[first_buy:]
