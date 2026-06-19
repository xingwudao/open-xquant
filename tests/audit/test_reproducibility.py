from __future__ import annotations

from decimal import Decimal

import pandas as pd

from oxq.audit.reproducibility import audit_reproducibility
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.spec.compiler import _write_artifacts
from oxq.spec.schema import StrategySpec


def test_reproducibility_audit_accepts_xshe_calendar_alias(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="xshe_audit", hypothesis="audit resolves aliased exchange calendars")
    spec.market.calendar = "XSHE"
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"], utc=True)
    source_df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [100, 100],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={"SPY": source_df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    source_df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "pass"
    assert any(check["id"] == "data_fingerprint" and check["status"] == "pass" for check in audit["checks"])
