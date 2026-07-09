import pandas as pd

from oxq.core.types import PortfolioOptimizer
from oxq.portfolio.optimizers import TopNRankingOptimizer


def test_protocol():
    assert isinstance(TopNRankingOptimizer(score_col="score"), PortfolioOptimizer)


def test_basic_ranking():
    opt = TopNRankingOptimizer(score_col="score", n=2)
    indicators = {
        "A": pd.DataFrame({"score": [30.0]}, index=pd.to_datetime(["2024-01-01"])),
        "B": pd.DataFrame({"score": [20.0]}, index=pd.to_datetime(["2024-01-01"])),
        "C": pd.DataFrame({"score": [10.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize({}, indicators)
    assert abs(result["A"] - 0.6) < 1e-9
    assert abs(result["B"] - 0.4) < 1e-9
    assert "C" not in result


def test_filter_negative():
    opt = TopNRankingOptimizer(score_col="score", n=5)
    indicators = {
        "A": pd.DataFrame({"score": [10.0]}, index=pd.to_datetime(["2024-01-01"])),
        "B": pd.DataFrame({"score": [-5.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize({}, indicators)
    assert result["A"] == 1.0
    assert "B" not in result


def test_max_weight_cap():
    opt = TopNRankingOptimizer(score_col="score", n=2, max_weight=0.5)
    indicators = {
        "A": pd.DataFrame({"score": [90.0]}, index=pd.to_datetime(["2024-01-01"])),
        "B": pd.DataFrame({"score": [10.0]}, index=pd.to_datetime(["2024-01-01"])),
    }
    result = opt.optimize({}, indicators)
    assert result["A"] == 0.5
    assert abs(result["B"] - 0.1) < 1e-9


def test_empty_returns_cash():
    opt = TopNRankingOptimizer(score_col="score", n=5)
    result = opt.optimize({}, {})
    assert result == {"CASH": 1.0}


def test_all_negative_returns_cash():
    opt = TopNRankingOptimizer(score_col="score", n=5)
    indicators = {"A": pd.DataFrame({"score": [-1.0]}, index=pd.to_datetime(["2024-01-01"]))}
    result = opt.optimize({}, indicators)
    assert result == {"CASH": 1.0}


def test_pre_filter_signal_selects_only_true_symbols_with_equal_weights():
    opt = TopNRankingOptimizer(
        score_col="score",
        n=2,
        pre_filter_signal="positive_momentum",
        weighting="equal",
    )
    dates = pd.to_datetime(["2024-01-01"])
    signals = {
        "A": pd.DataFrame({"positive_momentum": [True]}, index=dates),
        "B": pd.DataFrame({"positive_momentum": [False]}, index=dates),
        "C": pd.DataFrame({"positive_momentum": [True]}, index=dates),
    }
    indicators = {
        "A": pd.DataFrame({"score": [30.0]}, index=dates),
        "B": pd.DataFrame({"score": [100.0]}, index=dates),
        "C": pd.DataFrame({"score": [20.0]}, index=dates),
    }

    result = opt.optimize(signals, indicators)

    assert result == {"A": 0.5, "C": 0.5}


def test_pre_filter_signal_requires_positive_numeric_values():
    opt = TopNRankingOptimizer(
        score_col="score",
        n=2,
        pre_filter_signal="gate",
        weighting="equal",
        max_weight=0.5,
    )
    dates = pd.to_datetime(["2024-01-01"])
    signals = {
        "A": pd.DataFrame({"gate": [-1.0]}, index=dates),
        "B": pd.DataFrame({"gate": [1.0]}, index=dates),
    }
    indicators = {
        "A": pd.DataFrame({"score": [10.0]}, index=dates),
        "B": pd.DataFrame({"score": [9.0]}, index=dates),
    }

    result = opt.optimize(signals, indicators)

    assert result == {"B": 0.5, "CASH": 0.5}


def test_ascending_rank_selects_lowest_scores():
    opt = TopNRankingOptimizer(
        score_col="score",
        n=2,
        ascending=True,
        filter_negative=False,
        weighting="equal",
    )
    dates = pd.to_datetime(["2024-01-01"])
    indicators = {
        "A": pd.DataFrame({"score": [30.0]}, index=dates),
        "B": pd.DataFrame({"score": [10.0]}, index=dates),
        "C": pd.DataFrame({"score": [20.0]}, index=dates),
    }

    result = opt.optimize({}, indicators)

    assert result == {"B": 0.5, "C": 0.5}
