"""Module example: Factor Evaluation.

Demonstrates: IC, ICIR, Rank IC, decay, turnover, tearsheet,
multi-period comparison, and risk-adjusted momentum.

Run: uv run python examples/modules/08_factor_eval.py
"""

from pathlib import Path

import pandas as pd

from oxq.data.loaders import YFinanceDownloader
from oxq.data.market import LocalMarketDataProvider
from oxq.factor_eval.bundle import create_bundle
from oxq.factor_eval.metrics import compute_decay, compute_ic, compute_icir, compute_rank_ic, compute_turnover

SYMBOLS = ["SPY", "QQQ", "IWM"]
print(f"Downloading data for {SYMBOLS}...")
dloader = YFinanceDownloader()
for sym in SYMBOLS:
    dloader.download(symbol=sym, start="2018-01-01", end="2024-12-31")
market = LocalMarketDataProvider()

# ===========================================================================
# Part 1: Build factor (20-day momentum) and forward returns
# ===========================================================================
print("=" * 50)
print("FACTOR: 20-Day Momentum")
print("=" * 50)

# Build factor DataFrame (index=date, columns=symbol)
factor_df = pd.DataFrame()
prices_df = pd.DataFrame()
for sym in SYMBOLS:
    bars = market.get_bars(sym, "2018-01-01", "2024-12-31")
    close = bars["close"]
    prices_df[sym] = close
    factor_df[sym] = close.pct_change(20)  # 20-day momentum

# Drop NaN from lookback
factor_df = factor_df.dropna()
prices_df = prices_df.loc[factor_df.index]

# Forward returns: 1-day forward return
forward_returns = prices_df.pct_change(1).shift(-1)
common_idx = factor_df.index.intersection(forward_returns.index)
factor_df = factor_df.loc[common_idx]
forward_returns = forward_returns.loc[common_idx]
forward_returns = forward_returns.dropna()

# Trim to common dates
common_idx = factor_df.index.intersection(forward_returns.index)
factor_df = factor_df.loc[common_idx]
forward_returns = forward_returns.loc[common_idx]

print(f"Factor shape:  {factor_df.shape[0]} dates × {factor_df.shape[1]} symbols")
print(f"Forward shape: {forward_returns.shape}")

# ===========================================================================
# Part 2: IC (Information Coefficient)
# ===========================================================================
print(f"\n{'='*50}")
print("INFORMATION COEFFICIENT (IC)")
print("=" * 50)

ic_result = compute_ic(factor=factor_df, forward_returns=forward_returns)
print(f"  IC Mean:    {ic_result['mean']:.4f}")
print(f"  IC Std:     {ic_result['std']:.4f}")
print(f"  ICIR:       {compute_icir(ic_result['mean'], ic_result['std']):.4f}")
pos_days = sum(1 for v in ic_result["series"] if v > 0)
print(f"  IC > 0:     {pos_days} / {len(ic_result['series'])} ({pos_days/len(ic_result['series']):.1%})")

# ===========================================================================
# Part 3: Rank IC (Spearman)
# ===========================================================================
print(f"\n{'='*50}")
print("RANK IC (Spearman)")
print("=" * 50)

rank_ic = compute_rank_ic(factor=factor_df, forward_returns=forward_returns)
print(f"  Rank IC Mean: {rank_ic['mean']:.4f}")
print(f"  Rank IC Std:  {rank_ic['std']:.4f}")

# ===========================================================================
# Part 4: IC Decay
# ===========================================================================
print(f"\n{'='*50}")
print("IC DECAY (multi-horizon)")
print("=" * 50)

decay = compute_decay(
    factor=factor_df,
    prices=prices_df,
    horizons=[1, 3, 5, 10, 20],
)

print(f"\n{'Horizon':<10} {'IC Mean':<10}")
for h, ic_v in zip(decay["horizons"], decay["ic_values"]):
    print(f"  {h:<10} {ic_v:<10.4f}")

# ===========================================================================
# Part 5: Turnover
# ===========================================================================
print(f"\n{'='*50}")
print("FACTOR TURNOVER")
print("=" * 50)

to = compute_turnover(factor=factor_df)
print(f"  Mean turnover: {to:.4f}")
print("  (fraction of ranking changed per period; lower = more stable)")

# ===========================================================================
# Part 6: Multi-Period Comparison
# ===========================================================================
print(f"\n{'='*50}")
print("MULTI-PERIOD: 10d vs 30d vs 60d Momentum")
print("=" * 50)

periods = [10, 30, 60]
print(f"\n{'Period':<12} {'IC Mean':<10} {'ICIR':<10} {'Rank IC':<10} {'Turnover':<10}")
print("-" * 52)
for p in periods:
    factor_p = pd.DataFrame()
    for sym in SYMBOLS:
        bars = market.get_bars(sym, "2018-01-01", "2024-12-31")
        factor_p[sym] = bars["close"].pct_change(p)
    factor_p = factor_p.dropna()
    common_p = factor_p.index.intersection(forward_returns.dropna().index)
    factor_p = factor_p.loc[common_p]
    fwd_p = forward_returns.loc[common_p]

    ic_p = compute_ic(factor=factor_p, forward_returns=fwd_p)
    rk_p = compute_rank_ic(factor=factor_p, forward_returns=fwd_p)
    to_p = compute_turnover(factor=factor_p)
    icir_p = compute_icir(ic_p["mean"], ic_p["std"])
    print(f"  {p:<12} {ic_p['mean']:<10.4f} {icir_p:<10.4f} {rk_p['mean']:<10.4f} {to_p:<10.4f}")

# ===========================================================================
# Part 7: Risk-Adjusted Momentum
# ===========================================================================
print(f"\n{'='*50}")
print("RISK-ADJUSTED MOMENTUM")
print("=" * 50)

# Momentum / Volatility = signal strength adjusted for risk
ram_df = pd.DataFrame()
for sym in SYMBOLS:
    bars = market.get_bars(sym, "2018-01-01", "2024-12-31")
    close = bars["close"]
    mom = close.pct_change(20)
    vol = close.pct_change().rolling(20).std()
    ram_df[sym] = mom / vol.replace(0, float("nan"))
ram_df = ram_df.dropna()

common_ram = ram_df.index.intersection(forward_returns.dropna().index)
ram_df = ram_df.loc[common_ram]
fwd_ram = forward_returns.loc[common_ram]

print(f"\n{'':20} {'Momentum(20)':>16} {'RiskAdj-Mom(20)':>16}")
print("-" * 52)
mom_ic = compute_ic(factor=factor_df, forward_returns=forward_returns)
ram_ic = compute_ic(factor=ram_df, forward_returns=fwd_ram)
mom_icir = compute_icir(mom_ic["mean"], mom_ic["std"])
ram_icir = compute_icir(ram_ic["mean"], ram_ic["std"])
for label, m_key in [("IC Mean", "mean"), ("IC Std", "std"), ("ICIR", "")]:
    if m_key:
        print(f"  {label:<20} {mom_ic[m_key]:>16.4f} {ram_ic[m_key]:>16.4f}")
    else:
        print(f"  {label:<20} {mom_icir:>16.4f} {ram_icir:>16.4f}")

print("  (Risk-Adjusted = momentum / rolling volatility)")

# ===========================================================================
# Part 8: Tearsheet (visual summary)
# ===========================================================================
print(f"\n{'='*50}")
print("TEARSHEET")
print("=" * 50)

# Build a FactorBundle for tearsheet
factor_series = factor_df.stack().rename_axis(["date", "asset"]).rename("momentum_20")
bundle = create_bundle(
    factor_values=factor_series,
    prices=prices_df,
    forward_periods=[1],
)

try:
    from oxq.factor_eval.tearsheet import generate_tearsheet

    ts_dir = Path("/tmp/oxq_examples/tearsheet")
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_result = generate_tearsheet(
        bundle=bundle,
        forward_periods=[1],
        output_dir=str(ts_dir),
    )
    print("  Output: /tmp/oxq_examples/tearsheet/")
except ImportError:
    print("  (matplotlib not installed — skip tearsheet; install with: pip install open-xquant[chart])")

print("\nFactor evaluation complete.")
