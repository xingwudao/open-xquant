"""Session store — holds mutable state across tool calls.

State is persisted to a temp file so it survives subprocess restarts
(each tool call may run in a separate process).
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oxq.contrib.alpaca.market_data import AlpacaMarketDataProvider
    from oxq.core.strategy import Strategy
    from oxq.observe.audit import AuditRecord
    from oxq.observe.detector import MarketStateDetector
    from oxq.observe.experiment import ExperimentLog
    from oxq.observe.monitor import StrategyMonitor
    from oxq.optimize.paramset import ParameterSet
    from oxq.optimize.search import SearchResult
    from oxq.optimize.validation import CVResult
    from oxq.optimize.walk_forward import WalkForwardResult
    from oxq.portfolio.analytics import RunResult
    from oxq.trade.live_broker import LiveBroker
    from oxq.universe.base import UniverseProvider

logger = logging.getLogger(__name__)

_SESSION_FILE = Path(tempfile.gettempdir()) / "oxq_session.pkl"

_strategies: dict[str, Strategy] = {}
_strategy_universes: dict[str, UniverseProvider] = {}
_run_results: dict[str, RunResult] = {}
_paramsets: dict[str, ParameterSet] = {}
_search_results: dict[str, SearchResult] = {}
_wf_results: dict[str, WalkForwardResult] = {}
_cv_results: dict[str, CVResult] = {}
_monitors: dict[str, StrategyMonitor] = {}
_detectors: dict[str, MarketStateDetector] = {}
_experiment_logs: dict[str, ExperimentLog] = {}
_audit_records: dict[str, AuditRecord] = {}

# Live trading state (not persisted — requires fresh connect each session)
_live_broker: LiveBroker | None = None
_live_market: AlpacaMarketDataProvider | None = None


def _save() -> None:
    """Persist session state to disk."""
    try:
        with open(_SESSION_FILE, "wb") as f:
            pickle.dump(
                {
                    "strategies": _strategies,
                    "strategy_universes": _strategy_universes,
                    "run_results": _run_results,
                    "paramsets": _paramsets,
                    "search_results": _search_results,
                    "wf_results": _wf_results,
                    "cv_results": _cv_results,
                    "monitors": _monitors,
                    "detectors": _detectors,
                    "experiment_logs": _experiment_logs,
                    "audit_records": _audit_records,
                },
                f,
            )
    except Exception:
        logger.warning("Failed to save session state", exc_info=True)


def _load() -> None:
    """Load session state from disk (if available)."""
    if not _SESSION_FILE.exists():
        return
    try:
        with open(_SESSION_FILE, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        _strategies.update(data.get("strategies", {}))
        _strategy_universes.update(data.get("strategy_universes", {}))
        for name, strategy in _strategies.items():
            _migrate_legacy_strategy_rules(strategy)
            if name not in _strategy_universes:
                legacy_universe = _legacy_strategy_universe(strategy)
                if legacy_universe is not None:
                    _strategy_universes[name] = legacy_universe
        _run_results.update(data.get("run_results", {}))
        _paramsets.update(data.get("paramsets", {}))
        _search_results.update(data.get("search_results", {}))
        _wf_results.update(data.get("wf_results", {}))
        _cv_results.update(data.get("cv_results", {}))
        _monitors.update(data.get("monitors", {}))
        _detectors.update(data.get("detectors", {}))
        _experiment_logs.update(data.get("experiment_logs", {}))
        _audit_records.update(data.get("audit_records", {}))
    except Exception:
        logger.warning("Failed to load session state", exc_info=True)


def _migrate_legacy_strategy_rules(strategy: object) -> None:
    try:
        vars_dict = vars(strategy)
    except TypeError:
        return
    if "rules" in vars_dict:
        return
    pending_rules = vars_dict.get("_pending_rules", [])
    setattr(strategy, "rules", list(pending_rules or []))


def _legacy_strategy_universe(strategy: object) -> UniverseProvider | None:
    try:
        universe = getattr(strategy, "_legacy_universe", None)
    except Exception:
        universe = None
    if universe is None:
        try:
            universe = vars(strategy).get("universe")
        except TypeError:
            universe = None
    return universe if universe is not None and callable(getattr(universe, "get_universe", None)) else None


def clear() -> None:
    """Reset session state (for testing and Clear Chat)."""
    global _live_broker, _live_market
    _strategies.clear()
    _strategy_universes.clear()
    _run_results.clear()
    _paramsets.clear()
    _search_results.clear()
    _wf_results.clear()
    _cv_results.clear()
    _monitors.clear()
    _detectors.clear()
    _experiment_logs.clear()
    _audit_records.clear()
    if _live_broker is not None:
        _live_broker.close()
        _live_broker = None
    _live_market = None
    _SESSION_FILE.unlink(missing_ok=True)


# Auto-load persisted state when the tool process starts.
_load()
