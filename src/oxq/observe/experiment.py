"""Experiment log — structured iteration tracking for strategy development."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone; UTC = timezone.utc  # py3.9 compat
from typing import Any

import pandas as pd

from oxq.core.strategy import Strategy
from oxq.portfolio.analytics import RunResult

_METRIC_METHOD_MAP: dict[str, str] = {
    "total_return": "total_return",
    "sharpe": "sharpe_ratio",
    "sharpe_ratio": "sharpe_ratio",
    "max_drawdown": "max_drawdown",
    "annualized_return": "annualized_return",
    "annualized_volatility": "annualized_volatility",
    "calmar": "calmar_ratio",
    "calmar_ratio": "calmar_ratio",
    "sortino": "sortino_ratio",
    "sortino_ratio": "sortino_ratio",
}


@dataclass(frozen=True)
class Experiment:
    """A single experiment record."""

    name: str
    observation: str
    hypothesis: str
    criteria: dict[str, Any]
    result: dict[str, Any]
    conclusion: str
    notes: str
    timestamp: str
    run_id: str | None = None


class ExperimentLog:
    """Log for tracking strategy iteration experiments."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._experiments: list[Experiment] = []

    @property
    def experiments(self) -> list[Experiment]:
        return list(self._experiments)

    def add(
        self,
        name: str,
        observation: str,
        hypothesis: str,
        criteria: dict[str, Any],
        result: dict[str, Any],
        conclusion: str,
        notes: str = "",
        run_id: str | None = None,
    ) -> None:
        self._experiments.append(
            Experiment(
                name=name,
                observation=observation,
                hypothesis=hypothesis,
                criteria=criteria,
                result=result,
                conclusion=conclusion,
                notes=notes,
                timestamp=datetime.now(tz=UTC).isoformat(),
                run_id=run_id,
            ),
        )

    def add_from_strategy(
        self,
        strategy: Strategy,
        result: RunResult,
        observation: str,
        conclusion: str,
        notes: str = "",
        run_id: str | None = None,
    ) -> None:
        extracted: dict[str, Any] = {
            "total_return": result.total_return(),
            "sharpe_ratio": result.sharpe_ratio(),
            "max_drawdown": result.max_drawdown(),
            "annualized_return": result.annualized_return(),
            "currency": result.portfolio.currency,
        }

        for key in strategy.objectives:
            if key not in extracted:
                method_name = _METRIC_METHOD_MAP.get(key)
                if method_name and hasattr(result, method_name):
                    extracted[key] = getattr(result, method_name)()

        self.add(
            name=strategy.name,
            observation=observation,
            hypothesis=strategy.hypothesis,
            criteria=strategy.objectives,
            result=extracted,
            conclusion=conclusion,
            notes=notes,
            run_id=run_id,
        )

    def to_dataframe(self) -> pd.DataFrame:
        if not self._experiments:
            return pd.DataFrame()
        return pd.DataFrame([asdict(e) for e in self._experiments])

    def to_markdown(self) -> str:
        if not self._experiments:
            return ""
        try:
            import tabulate as _  # noqa: F401
        except ImportError:
            msg = "to_markdown() requires the 'tabulate' package: pip install tabulate"
            raise ImportError(msg) from None
        df = self.to_dataframe()
        return df.to_markdown(index=False)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "experiments": [asdict(e) for e in self._experiments],
        }

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentLog:
        log = cls(name=d.get("name", ""))
        for exp_data in d.get("experiments", []):
            log._experiments.append(Experiment(**exp_data))
        return log
