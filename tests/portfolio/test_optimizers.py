import pandas as pd
import pytest

from oxq.core.types import PortfolioOptimizer
from oxq.portfolio.optimizers import (
    EqualWeightOptimizer,
    KellyOptimizer,
    PctEquityOptimizer,
    RiskParityOptimizer,
    TopNRankingOptimizer,
)


def test_equal_weight_protocol():
    assert isinstance(EqualWeightOptimizer(), PortfolioOptimizer)


def test_equal_weight_basic():
    opt = EqualWeightOptimizer()
    signals = {
        "AAPL": pd.DataFrame({"signal": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
        "GOOG": pd.DataFrame({"signal": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize(signals, {})
    assert result["AAPL"] == 0.5
    assert result["GOOG"] == 0.5


def test_equal_weight_single_asset():
    opt = EqualWeightOptimizer()
    signals = {"AAPL": pd.DataFrame({"signal": [1.0]}, index=pd.to_datetime(["2024-01-01"]))}
    result = opt.optimize(signals, {})
    assert result["AAPL"] == 1.0


def test_equal_weight_empty():
    opt = EqualWeightOptimizer()
    result = opt.optimize({}, {})
    assert result == {"CASH": 1.0}


def test_risk_parity_protocol():
    assert isinstance(RiskParityOptimizer(volatility_col="vol"), PortfolioOptimizer)


def test_risk_parity_basic():
    opt = RiskParityOptimizer(volatility_col="volatility")
    indicators = {
        "AAPL": pd.DataFrame({"volatility": [0.1]}, index=pd.to_datetime(["2024-01-01"])),
        "GOOG": pd.DataFrame({"volatility": [0.2]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize({}, indicators)
    assert abs(result["AAPL"] - 2/3) < 1e-9
    assert abs(result["GOOG"] - 1/3) < 1e-9


def test_risk_parity_zero_volatility_excluded():
    opt = RiskParityOptimizer(volatility_col="volatility")
    indicators = {
        "AAPL": pd.DataFrame({"volatility": [0.1]}, index=pd.to_datetime(["2024-01-01"])),
        "GOOG": pd.DataFrame({"volatility": [0.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize({}, indicators)
    assert result["AAPL"] == 1.0
    assert "GOOG" not in result


def test_kelly_protocol():
    assert isinstance(KellyOptimizer(win_rate_col="wr", avg_win_col="aw", avg_loss_col="al"), PortfolioOptimizer)


def test_kelly_basic():
    opt = KellyOptimizer(win_rate_col="win_rate", avg_win_col="avg_win", avg_loss_col="avg_loss")
    indicators = {
        "AAPL": pd.DataFrame({"win_rate": [0.6], "avg_win": [2.0], "avg_loss": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize({}, indicators)
    assert abs(result["AAPL"] - 0.4) < 1e-9
    assert abs(result.get("CASH", 0) - 0.6) < 1e-9


def test_kelly_fractional():
    opt = KellyOptimizer(win_rate_col="win_rate", avg_win_col="avg_win", avg_loss_col="avg_loss", fraction=0.5)
    indicators = {
        "AAPL": pd.DataFrame({"win_rate": [0.6], "avg_win": [2.0], "avg_loss": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize({}, indicators)
    assert abs(result["AAPL"] - 0.2) < 1e-9


def test_kelly_negative_edge_goes_to_cash():
    opt = KellyOptimizer(win_rate_col="win_rate", avg_win_col="avg_win", avg_loss_col="avg_loss")
    indicators = {
        "AAPL": pd.DataFrame({"win_rate": [0.3], "avg_win": [1.0], "avg_loss": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize({}, indicators)
    assert result.get("AAPL", 0) == 0
    assert result["CASH"] == 1.0


def test_pct_equity_protocol():
    assert isinstance(PctEquityOptimizer(), PortfolioOptimizer)


def test_pct_equity_basic():
    """Each signaled symbol gets pct_equity weight, rest goes to CASH."""
    opt = PctEquityOptimizer(pct=0.10)
    signals = {
        "AAPL": pd.DataFrame({"signal": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
        "GOOG": pd.DataFrame({"signal": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize(signals, {})
    assert abs(result["AAPL"] - 0.10) < 1e-9
    assert abs(result["GOOG"] - 0.10) < 1e-9
    assert abs(result["CASH"] - 0.80) < 1e-9


def test_pct_equity_exceeds_one():
    """When pct * n_symbols > 1.0, normalize to sum to 1.0."""
    opt = PctEquityOptimizer(pct=0.40)
    signals = {
        "AAPL": pd.DataFrame({"signal": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
        "GOOG": pd.DataFrame({"signal": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
        "MSFT": pd.DataFrame({"signal": [1.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize(signals, {})
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-9
    for sym in ["AAPL", "GOOG", "MSFT"]:
        assert abs(result[sym] - 1.0 / 3) < 1e-9


def test_pct_equity_empty():
    opt = PctEquityOptimizer(pct=0.10)
    result = opt.optimize({}, {})
    assert result == {"CASH": 1.0}


class TestTopNRankingOptimizerRedistribution:
    def test_weights_sum_to_one_after_cap(self):
        """After capping at max_weight, weights must sum to 1.0."""
        opt = TopNRankingOptimizer(score_col="score", n=3, max_weight=0.5)
        indicators = {
            "A": pd.DataFrame({"score": [0.8]}),
            "B": pd.DataFrame({"score": [0.15]}),
            "C": pd.DataFrame({"score": [0.05]}),
        }
        result = opt.optimize({}, indicators)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_excess_goes_to_cash(self):
        """Excess weight from capping goes to CASH."""
        opt = TopNRankingOptimizer(score_col="score", n=3, max_weight=0.5)
        indicators = {
            "A": pd.DataFrame({"score": [0.8]}),
            "B": pd.DataFrame({"score": [0.15]}),
            "C": pd.DataFrame({"score": [0.05]}),
        }
        result = opt.optimize({}, indicators)
        # A: 0.8/1.0 = 0.8, capped to 0.5, excess 0.3 -> CASH
        assert result["A"] == pytest.approx(0.5)
        assert "CASH" in result
        assert result["CASH"] == pytest.approx(0.3)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_no_cap_no_cash(self):
        """When no weight exceeds max, no CASH key should be added."""
        opt = TopNRankingOptimizer(score_col="score", n=3, max_weight=0.9)
        indicators = {
            "A": pd.DataFrame({"score": [0.5]}),
            "B": pd.DataFrame({"score": [0.5]}),
        }
        result = opt.optimize({}, indicators)
        assert "CASH" not in result
        assert sum(result.values()) == pytest.approx(1.0)

    def test_score_weighting_does_not_emit_negative_target_weights(self):
        opt = TopNRankingOptimizer(score_col="score", n=2, weighting="score", filter_negative=False)
        indicators = {
            "A": pd.DataFrame({"score": [1.0]}),
            "B": pd.DataFrame({"score": [-1.0]}),
        }

        result = opt.optimize({}, indicators)

        assert result == {"CASH": 1.0}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_col": ""},
        {"n": 0},
        {"n": True},
        {"max_weight": -0.5},
        {"max_weight": 1.5},
        {"max_weight": float("nan")},
        {"pre_filter_signal": 1},
        {"filter_negative": "yes"},
        {"ascending": "no"},
        {"weighting": 1},
        {"weighting": "invalid"},
    ],
)
def test_top_n_ranking_rejects_invalid_constructor_params(kwargs: dict[str, object]) -> None:
    params = {"score_col": "score", **kwargs}

    with pytest.raises(ValueError):
        TopNRankingOptimizer(**params)
