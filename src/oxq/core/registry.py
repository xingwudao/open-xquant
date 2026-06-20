"""Component registry — single source of truth for Indicators, Signals, Optimizers, and Rules.

All built-in components are registered at import time via ``_load_builtins()``.
Third-party packages can register additional components via the public
``register_indicator``, ``register_signal``, ``register_portfolio_optimizer``,
and ``register_rule`` functions.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
from typing import Any

from oxq.core.types import Indicator, PortfolioOptimizer, Rule, Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Private registry dicts: name -> class
# ---------------------------------------------------------------------------

_INDICATOR_REGISTRY: dict[str, type] = {}
_SIGNAL_REGISTRY: dict[str, type] = {}
_PORTFOLIO_OPTIMIZER_REGISTRY: dict[str, type] = {}
_RULE_REGISTRY: dict[str, type] = {}

# Indicator metadata: name -> {description, category, source_type, ...}
_INDICATOR_METADATA: dict[str, dict[str, str]] = {}


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _register(cls: type, protocol: type, registry: dict[str, type]) -> None:
    """Validate *cls* against *protocol* and store it in *registry*.

    Uses ``inspect.signature`` to determine whether the constructor requires
    arguments.  If it does, validation falls back to a structural check so
    that real ``TypeError``s raised during instantiation are never swallowed.
    """
    sig = inspect.signature(cls.__init__)
    required_params = [
        p
        for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    if required_params:
        # Cannot instantiate without arguments — structural check only
        _check_class_structure(cls, protocol)
    else:
        instance = cls()
        if not isinstance(instance, protocol):
            msg = f"{cls.__name__} does not satisfy {protocol.__name__} protocol"
            raise TypeError(msg)

    name: str = getattr(cls, "name", cls.__name__)
    registry[name] = cls


def _check_class_structure(cls: type, protocol: type) -> None:
    """Verify *cls* has the methods/attributes declared by *protocol*."""
    # Collect required members from the protocol (skip dunder and private)
    hints: dict[str, Any] = {}
    for base in protocol.__mro__:
        hints.update(getattr(base, "__annotations__", {}))

    for attr_name in hints:
        if attr_name.startswith("_"):
            continue
        if not hasattr(cls, attr_name):
            msg = f"{cls.__name__} does not satisfy {protocol.__name__} protocol"
            raise TypeError(msg)

    # Check callable members from the protocol (e.g. compute, optimize, evaluate)
    for member_name in dir(protocol):
        if member_name.startswith("_"):
            continue
        proto_member = getattr(protocol, member_name, None)
        if callable(proto_member) and not isinstance(proto_member, type):
            if not hasattr(cls, member_name) or not callable(
                getattr(cls, member_name, None)
            ):
                msg = (
                    f"{cls.__name__} does not satisfy {protocol.__name__} protocol"
                )
                raise TypeError(msg)


# ---------------------------------------------------------------------------
# Public register functions
# ---------------------------------------------------------------------------

def register_indicator(cls: type) -> None:
    """Register an Indicator class."""
    _register(cls, Indicator, _INDICATOR_REGISTRY)


def register_signal(cls: type) -> None:
    """Register a Signal class."""
    _register(cls, Signal, _SIGNAL_REGISTRY)


def register_portfolio_optimizer(cls: type) -> None:
    """Register a PortfolioOptimizer class."""
    _register(cls, PortfolioOptimizer, _PORTFOLIO_OPTIMIZER_REGISTRY)


def register_rule(cls: type) -> None:
    """Register a Rule class."""
    _register(cls, Rule, _RULE_REGISTRY)


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def list_indicators() -> dict[str, type]:
    """Return a copy of the indicator registry."""
    return dict(_INDICATOR_REGISTRY)


def list_signals() -> dict[str, type]:
    """Return a copy of the signal registry."""
    return dict(_SIGNAL_REGISTRY)


def list_portfolio_optimizers() -> dict[str, type]:
    """Return a copy of the portfolio optimizer registry."""
    return dict(_PORTFOLIO_OPTIMIZER_REGISTRY)


def list_rules() -> dict[str, type]:
    """Return a copy of the rule registry."""
    return dict(_RULE_REGISTRY)


def register_indicator_metadata(
    name: str,
    description: str,
    category: str,
    source_type: str,
    display_name: str = "",
    value_range: str = "",
    typical_usage: str = "",
) -> None:
    """Register metadata for an indicator (for Agent understanding)."""
    _INDICATOR_METADATA[name] = {
        "description": description,
        "category": category,
        "source_type": source_type,
        "display_name": display_name or name,
        "value_range": value_range,
        "typical_usage": typical_usage,
    }


def get_indicator_metadata(name: str) -> dict[str, str] | None:
    """Return metadata for a single indicator, or None."""
    return _INDICATOR_METADATA.get(name)


def list_indicator_metadata() -> dict[str, dict[str, str]]:
    """Return a copy of all indicator metadata."""
    return dict(_INDICATOR_METADATA)


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------

def _load_builtins() -> None:
    """Import and register all built-in components."""

    # -- Indicators ----------------------------------------------------------
    from oxq.indicators import (
        ADX,
        AROON,
        ATR,
        BP,
        CCI,
        DEMA,
        EMA,
        EP,
        MFI,
        OBV,
        PB,
        PE,
        PPO,
        ROC,
        RSI,
        SMA,
        TEMA,
        VWAP,
        WMA,
        AccrualRatio,
        AnnualizedVolatility,
        BollingerLower,
        BollingerUpper,
        CashFlowRatio,
        GarchVolatility,
        HurstExponent,
        IchimokuChikou,
        IchimokuKijun,
        IchimokuSenkouA,
        IchimokuSenkouB,
        IchimokuTenkan,
        LogReturn,
        MACDHistogram,
        MACDLine,
        MACDSignal,
        MarketCap,
        Momentum,
        NdayReturn,
        NetProfitMargin,
        PowerRatio,
        Ratio,
        ROEChange,
        RollingMDD,
        RollingVolatility,
        SimpleMomentum,
        StochK,
        TurnoverRate,
    )

    for cls in (
        ADX, AROON, ATR, AccrualRatio, AnnualizedVolatility, BP,
        BollingerLower, BollingerUpper, CCI, CashFlowRatio, DEMA, EMA, EP,
        GarchVolatility, HurstExponent, IchimokuChikou, IchimokuKijun,
        IchimokuSenkouA, IchimokuSenkouB, IchimokuTenkan, LogReturn,
        MACDHistogram, MACDLine, MACDSignal, MFI, MarketCap, Momentum,
        NdayReturn, NetProfitMargin, OBV, PB, PE, PPO, PowerRatio, ROC,
        ROEChange, RSI, Ratio, RollingMDD, RollingVolatility, SMA,
        SimpleMomentum, StochK, TEMA, TurnoverRate, VWAP, WMA,
    ):
        _register(cls, Indicator, _INDICATOR_REGISTRY)

    # -- Signals -------------------------------------------------------------
    from oxq.signals import (
        Comparison,
        Composite,
        Crossover,
        Formula,
        Peak,
        ROCTiming,
        Threshold,
        Timestamp,
    )

    for cls in (Comparison, Composite, Crossover, Formula, Peak, ROCTiming, Threshold, Timestamp):
        _register(cls, Signal, _SIGNAL_REGISTRY)

    # -- Portfolio Optimizers ------------------------------------------------
    from oxq.portfolio.optimizers import (
        EqualWeightOptimizer,
        KellyOptimizer,
        PctEquityOptimizer,
        RiskParityOptimizer,
        SignalToPositionOptimizer,
        TopNRankingOptimizer,
    )

    for cls in (
        EqualWeightOptimizer,
        RiskParityOptimizer,
        KellyOptimizer,
        TopNRankingOptimizer,
        PctEquityOptimizer,
        SignalToPositionOptimizer,
    ):
        _register(cls, PortfolioOptimizer, _PORTFOLIO_OPTIMIZER_REGISTRY)

    # -- Rules ---------------------------------------------------------------
    from oxq.rules import (
        BlacklistRule,
        DailyLossLimitRisk,
        ExitRule,
        MaxDrawdownRisk,
        MaxHoldingsRule,
        RebalanceFrequencyRule,
        StopLossRule,
        TakeProfitRule,
        TrailingStopRule,
    )

    for cls in (
        BlacklistRule,
        DailyLossLimitRisk,
        ExitRule,
        MaxDrawdownRisk,
        MaxHoldingsRule,
        RebalanceFrequencyRule,
        StopLossRule,
        TakeProfitRule,
        TrailingStopRule,
    ):
        _register(cls, Rule, _RULE_REGISTRY)

    # -- Indicator Metadata ---------------------------------------------------
    _metadata = [
        # Trend
        ("SMA", "Simple Moving Average of closing prices.", "trend", "compute"),
        ("EMA", "Exponential Moving Average, weights recent prices more.", "trend", "compute"),
        ("WMA", "Weighted Moving Average.", "trend", "compute"),
        ("DEMA", "Double Exponential Moving Average.", "trend", "compute"),
        ("TEMA", "Triple Exponential Moving Average.", "trend", "compute"),
        ("IchimokuTenkan", "Ichimoku Tenkan-sen (conversion line).", "trend", "compute"),
        ("IchimokuKijun", "Ichimoku Kijun-sen (base line).", "trend", "compute"),
        ("IchimokuSenkouA", "Ichimoku Senkou Span A (leading span A).", "trend", "compute"),
        ("IchimokuSenkouB", "Ichimoku Senkou Span B (leading span B).", "trend", "compute"),
        ("IchimokuChikou", "Ichimoku Chikou Span (lagging span).", "trend", "compute"),
        # Momentum
        ("RSI", "Relative Strength Index, 0-100 oscillator.", "momentum", "compute"),
        ("ROC", "Rate of Change, percentage price change.", "momentum", "compute"),
        ("PPO", "Percentage Price Oscillator.", "momentum", "compute"),
        ("Momentum", "Absolute price change over N periods.", "momentum", "compute"),
        ("NdayReturn", "N-day percentage return.", "momentum", "compute"),
        ("LogReturn", "Logarithmic return.", "momentum", "compute"),
        ("SimpleMomentum", "Simple momentum factor.", "momentum", "compute"),
        ("StochK", "Stochastic %K oscillator.", "momentum", "compute"),
        # MACD
        ("MACDLine", "MACD line (fast EMA - slow EMA).", "macd", "compute"),
        ("MACDSignal", "MACD signal line.", "macd", "compute"),
        ("MACDHistogram", "MACD histogram (MACD - signal).", "macd", "compute"),
        # Volatility
        ("BollingerUpper", "Upper Bollinger Band.", "volatility", "compute"),
        ("BollingerLower", "Lower Bollinger Band.", "volatility", "compute"),
        ("ATR", "Average True Range.", "volatility", "compute"),
        ("RollingVolatility", "Rolling standard deviation of returns.", "volatility", "compute"),
        ("RollingMDD", "Rolling Maximum Drawdown.", "volatility", "compute"),
        ("AnnualizedVolatility", "Annualized rolling volatility.", "volatility", "compute"),
        ("GarchVolatility", "GARCH(1,1) conditional volatility.", "volatility", "compute"),
        ("HurstExponent", "Hurst exponent for trend persistence.", "volatility", "compute"),
        # Volume
        ("OBV", "On-Balance Volume.", "volume", "compute"),
        ("VWAP", "Volume Weighted Average Price.", "volume", "compute"),
        ("MFI", "Money Flow Index.", "volume", "compute"),
        ("TurnoverRate", "Turnover rate.", "volume", "compute"),
        # Direction
        ("ADX", "Average Directional Index, trend strength 0-100.", "direction", "compute"),
        ("AROON", "Aroon indicator.", "direction", "compute"),
        ("CCI", "Commodity Channel Index.", "direction", "compute"),
        # Valuation
        ("PE", "Price-to-Earnings ratio (price / EPS).", "valuation", "compute"),
        ("PB", "Price-to-Book ratio (price / book value per share).", "valuation", "compute"),
        ("EP", "Earnings-to-Price ratio (EPS / price).", "valuation", "compute"),
        ("BP", "Book-to-Price ratio (book value / price).", "valuation", "compute"),
        ("MarketCap", "Market capitalization.", "valuation", "compute"),
        # Quality
        ("AccrualRatio", "Accrual ratio.", "quality", "compute"),
        ("CashFlowRatio", "Cash flow ratio.", "quality", "compute"),
        ("NetProfitMargin", "Net profit margin.", "quality", "compute"),
        ("ROEChange", "Change in ROE.", "quality", "compute"),
        # Factor
        ("Ratio", "Generic ratio between two columns.", "factor", "compute"),
        ("PowerRatio", "Power ratio indicator.", "factor", "compute"),
    ]

    for name, desc, cat, src in _metadata:
        register_indicator_metadata(name, desc, cat, src)


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------

def _entry_points() -> dict[str, list[Any]]:
    """Wrapper for importlib.metadata.entry_points (mockable in tests)."""
    result: dict[str, list[Any]] = {}
    for group in ("oxq.indicators", "oxq.signals", "oxq.portfolio_optimizers", "oxq.rules"):
        eps = importlib.metadata.entry_points(group=group)
        if eps:
            result[group] = list(eps)
    return result


_EP_GROUP_TO_REGISTER = {
    "oxq.indicators": register_indicator,
    "oxq.signals": register_signal,
    "oxq.portfolio_optimizers": register_portfolio_optimizer,
    "oxq.rules": register_rule,
}


def _load_entry_points() -> None:
    """Discover and register components from installed entry points."""
    for group, eps in _entry_points().items():
        register_fn = _EP_GROUP_TO_REGISTER.get(group)
        if not register_fn:
            continue
        for ep in eps:
            try:
                cls = ep.load()
                register_fn(cls)
            except Exception:
                logger.warning(
                    "Failed to load entry point %s from group %s",
                    ep.name,
                    group,
                    exc_info=True,
                )


# Register built-ins at module load time, then discover entry points.
_load_builtins()
_load_entry_points()
