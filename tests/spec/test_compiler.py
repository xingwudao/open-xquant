from __future__ import annotations

import hashlib
import json
from decimal import Decimal

import pandas as pd
import pytest

import oxq.audit.reproducibility as reproducibility
import oxq.spec.compiler as compiler
from oxq.audit.reproducibility import _hash_json_file, audit_reproducibility
from oxq.core.engine import Engine
from oxq.core.types import Order, Portfolio
from oxq.portfolio.analytics import RunResult
from oxq.portfolio.orderbook import ManagedOrder
from oxq.spec.compiler import _build_metrics, _build_optimizer, _write_artifacts, compile_run, compile_strategy
from oxq.spec.schema import IndicatorDef, SignalRuleDef, StrategySpec


def test_artifact_spec_hash_matches_serialized_spec(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="hash_test", hypothesis="hash artifacts are reproducible")
    spec.execution.initial_cash = 100_000

    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result.mktdata["SPY"].to_parquet(data_dir / "SPY.parquet")

    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "pass"


def test_reproducibility_audit_allows_self_contained_run_without_parent_digest(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="self_contained", hypothesis="run directories are portable")
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result.mktdata["SPY"].to_parquet(data_dir / "SPY.parquet")

    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    (tmp_path.parent / "run_digests.jsonl").unlink()

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "pass"
    assert not any(check["id"] == "run_digest" for check in audit["checks"])


def test_reproducibility_audit_fails_when_parent_digest_mismatches(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="digest_mismatch", hypothesis="external digest detects artifact hash tampering")
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path.parent / "run_digests.jsonl").write_text(
        json.dumps({"run_id": tmp_path.name, "artifact_hashes": "sha256:badbadbadbadbadb"}) + "\n",
        encoding="utf-8",
    )

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    digest_check = next(check for check in audit["checks"] if check["id"] == "run_digest")
    assert digest_check["status"] == "fail"
    assert digest_check["severity"] == "fatal"


def test_reproducibility_audit_fails_when_parent_digest_is_corrupt(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="digest_corrupt", hypothesis="external digest corruption is fatal")
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path.parent / "run_digests.jsonl").write_text("{not-json\n", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    digest_check = next(check for check in audit["checks"] if check["id"] == "run_digest")
    assert audit["status"] == "fail"
    assert digest_check["severity"] == "fatal"


def test_artifacts_persist_full_order_lifecycle_and_hash(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="order_lifecycle", hypothesis="orders are auditable even without fills")
    dates = pd.bdate_range("2024-01-02", periods=2, tz="UTC")
    rejected = ManagedOrder(
        Order(symbol="SPY", side="BUY", shares=10, order_type="market"),
        id="ord_1",
        created_at=dates[0].isoformat(),
    )
    rejected.status = "rejected"
    rejected.due_at = dates[1].isoformat()
    expired = ManagedOrder(
        Order(symbol="QQQ", side="BUY", shares=5, order_type="market"),
        id="ord_2",
        created_at=dates[0].isoformat(),
    )
    expired.status = "expired"
    expired.due_at = dates[1].isoformat()
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100000.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {"open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "volume": [1, 1]},
                index=dates,
            )
        },
        orders=[rejected, expired],
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    orders = pd.read_csv(tmp_path / "orders.csv")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert set(orders["status"]) == {"rejected", "expired"}
    assert set(["id", "created_at", "due_at", "status"]).issubset(orders.columns)
    assert "orders.csv" in hashes


def test_missing_ratio_ignores_derived_indicator_nans(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_ratio", hypothesis="derived warmup nans are not raw data missing")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                    "sma_50": [None, None, 1.0],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_counts_absent_required_columns(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_required", hypothesis="absent required columns are missing data")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0


def test_missing_ratio_counts_empty_symbol_as_fully_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="empty_symbol", hypothesis="empty market data is missing data")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[],
        mktdata={
            "SPY": pd.DataFrame(columns=spec.data.required_columns),
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 1.0


def test_missing_ratio_counts_sparse_symbol_calendar(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="sparse_calendar", hypothesis="missing symbol days are data gaps")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            ),
            "QQQ": pd.DataFrame(
                {
                    "open": [1.0, 1.0],
                    "high": [1.0, 1.0],
                    "low": [1.0, 1.0],
                    "close": [1.0, 1.0],
                    "volume": [1, 1],
                },
                index=[dates[0], dates[2]],
            ),
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0
    assert manifest["symbol_ranges"]["QQQ"]["start"] == "2024-01-01"
    assert manifest["symbol_ranges"]["QQQ"]["end"] == "2024-01-03"


def test_missing_ratio_counts_full_market_missing_business_day(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="full_gap", hypothesis="full-market gaps count as missing data")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    dates = pd.to_datetime(["2024-01-01", "2024-01-03"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[
            (dates[0], 100000.0),
            (pd.Timestamp("2024-01-02", tz="UTC"), 100000.0),
            (dates[1], 100001.0),
        ],
        mktdata={"SPY": frame, "QQQ": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0


def test_missing_ratio_does_not_count_xnys_holiday_as_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="holiday_gap", hypothesis="exchange holidays are not missing data")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-07-03", "2024-07-05"]
    dates = pd.to_datetime(["2024-07-03", "2024-07-05"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[
            (dates[0], 100000.0),
            (pd.Timestamp("2024-07-04", tz="UTC"), 100000.0),
            (dates[1], 100001.0),
        ],
        mktdata={"SPY": frame, "QQQ": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_does_not_count_xnys_special_closure(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="special_closure", hypothesis="special exchange closures are not missing data")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2018-12-04", "2018-12-06"]
    dates = pd.to_datetime(["2018-12-04", "2018-12-06"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={"SPY": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_requires_exchange_calendar_when_sessions_are_unavailable(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(strategy_id="calendar_required", hypothesis="incomplete fallback calendars are unsafe")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2018-12-04", "2018-12-06"]
    dates = pd.to_datetime(["2018-12-04", "2018-12-06"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0)],
        mktdata={"SPY": frame},
    )
    monkeypatch.setattr(compiler, "_exchange_calendar_sessions", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="exchange_calendars is required"):
        _write_artifacts(spec, result, tmp_path, Engine())


def test_missing_ratio_does_not_count_xnys_good_friday_as_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="good_friday_gap", hypothesis="exchange-only holidays are not missing")
    spec.market.calendar = "XNYS"
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-03-28", "2024-04-01"]
    dates = pd.to_datetime(["2024-03-28", "2024-04-01"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[
            (dates[0], 100000.0),
            (pd.Timestamp("2024-03-29", tz="UTC"), 100000.0),
            (dates[1], 100001.0),
        ],
        mktdata={"SPY": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] == 0.0


def test_missing_ratio_counts_missing_requested_start_boundary(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="start_boundary_gap", hypothesis="requested boundary gaps count")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-05"]
    dates = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"], utc=True)
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(date, 100000.0 + idx) for idx, date in enumerate(dates)],
        mktdata={"SPY": frame},
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_ratio"] > 0.0


def test_reproducibility_audit_fails_when_source_data_changes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="source_data_hash", hypothesis="source data changes break reproducibility")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_artifacts(spec, result, run_dir, Engine(), effective_data_dir=str(data_dir))

    tampered = df.copy()
    tampered.loc[dates[1], "close"] = 2.0
    tampered.to_parquet(data_dir / "SPY.parquet")

    audit = audit_reproducibility(run_dir)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_fingerprint" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_normalizes_naive_source_data_index(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="naive_source_hash", hypothesis="provider localizes naive source index")
    naive_dates = pd.bdate_range("2024-01-02", periods=3)
    aware_dates = naive_dates.tz_localize("UTC")
    source_df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=naive_dates,
    )
    loaded_df = source_df.copy()
    loaded_df.index = aware_dates
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(aware_dates[0], 100000.0), (aware_dates[1], 100001.0), (aware_dates[2], 100003.0)],
        mktdata={"SPY": loaded_df},
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


def test_reproducibility_audit_filters_source_data_to_manifest_calendar(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="calendar_source_hash", hypothesis="audit uses runtime calendar filter")
    dates = pd.to_datetime(["2024-01-02", "2024-01-06"], utc=True)
    source_df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [1, 1],
        },
        index=dates,
    )
    loaded_df = source_df.iloc[[0]]
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0)],
        mktdata={"SPY": loaded_df},
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


def test_reproducibility_calendar_alignment_accepts_aware_sessions(monkeypatch) -> None:
    class _AwareCalendar:
        @staticmethod
        def sessions_in_range(start, end):
            return pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")

    import exchange_calendars as xcals

    monkeypatch.setattr(xcals, "get_calendar", lambda _name: _AwareCalendar())
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=index)

    aligned = reproducibility._align_to_calendar_sessions(df, "XNYS", "2024-01-02", "2024-01-03")

    assert aligned.index.equals(index)


def test_reproducibility_audit_warns_when_source_data_dir_unavailable(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_data_dir", hypothesis="source fingerprints need source data")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())

    audit = audit_reproducibility(tmp_path)

    check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert check["status"] == "fail"
    assert check["severity"] == "fatal"
    assert audit["status"] == "fail"


def test_compile_run_rejects_strategy_id_path_traversal(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="../outside", hypothesis="strategy ids cannot be paths")

    with pytest.raises(ValueError, match="strategy_id"):
        compile_run(spec, out_dir=tmp_path)


def test_compile_run_rejects_invalid_specs_before_execution(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="invalid_costs", hypothesis="compile_run enforces validation")
    spec.cost.fee_rate = 0.0
    spec.cost.slippage_rate = 0.0

    with pytest.raises(ValueError, match="Spec validation failed"):
        compile_run(spec, out_dir=tmp_path)


def test_compile_run_records_resolved_default_data_dir(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(strategy_id="resolved_data_dir", hypothesis="default data dir is auditable")
    market_dir = tmp_path / "oxq_data" / "market"
    market_dir.mkdir(parents=True)
    dates = pd.bdate_range("2022-01-03", periods=3, tz="UTC")
    pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    ).to_parquet(market_dir / "SPY.parquet")
    monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path / "oxq_data"))

    _, run_dir = compile_run(spec, out_dir=tmp_path / "runs")

    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert env["data_dir"] == str(market_dir)
    assert audit_reproducibility(run_dir)["status"] == "pass"


def test_compile_run_records_absolute_data_dir_for_relative_input(tmp_path, monkeypatch) -> None:
    spec = StrategySpec.template(strategy_id="relative_data_dir", hypothesis="artifact data_dir is cwd-independent")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.bdate_range("2022-01-03", periods=3, tz="UTC")
    pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")
    monkeypatch.chdir(tmp_path)

    _, run_dir = compile_run(spec, data_dir="data", out_dir=tmp_path / "runs")

    env = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert env["data_dir"] == str(data_dir.resolve())


def test_compile_run_uses_spec_market_currency(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="usd_currency", hypothesis="spec currency controls portfolio currency")
    spec.market.currency = "USD"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.bdate_range("2022-01-03", periods=3, tz="UTC")
    pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")

    result, _ = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    assert result.portfolio.currency == "USD"
    assert {fill.order.currency for fill in result.trades} <= {"USD"}


def test_compile_run_expires_pending_buy_for_latched_sparse_symbol_missing_next_open(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="sparse_latch", hypothesis="latched sparse symbols do not duplicate buys")
    spec.universe.symbols = ["AAA", "BBB"]
    spec.benchmark.symbols = []
    spec.data.min_start_date = "2024-01-01"
    spec.data.required_columns = ["open", "high", "low", "close", "volume", "fast", "slow"]
    spec.signal.rules = {"entry": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"})}
    spec.validation.train_period = ["2024-01-01", "2024-01-01"]
    spec.validation.test_period = ["2024-01-02", "2024-01-05"]
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    aaa_dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"], utc=True)
    pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0, 10.0],
            "volume": [100, 100, 100, 100],
            "fast": [0.0, 2.0, 2.0, 2.0],
            "slow": [1.0, 1.0, 1.0, 1.0],
        },
        index=aaa_dates,
    ).to_parquet(data_dir / "AAA.parquet")
    bbb_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"], utc=True)
    pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [100, 100, 100, 100],
            "fast": [0.0, 0.0, 0.0, 0.0],
            "slow": [1.0, 1.0, 1.0, 1.0],
        },
        index=bbb_dates,
    ).to_parquet(data_dir / "BBB.parquet")

    result, _ = compile_run(spec, data_dir=str(data_dir), out_dir=tmp_path / "runs")

    buy_fills = [fill for fill in result.trades if fill.order.symbol == "AAA" and fill.order.side == "BUY"]
    assert len(buy_fills) == 0


def test_compile_run_rejects_multiple_crossover_rules(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="multi_cross", hypothesis="each crossover has an exit")
    spec.signal.indicators = {
        "fast_a": IndicatorDef(type="SMA", params={"period": 2}),
        "slow_a": IndicatorDef(type="SMA", params={"period": 3}),
        "fast_b": IndicatorDef(type="SMA", params={"period": 4}),
        "slow_b": IndicatorDef(type="SMA", params={"period": 5}),
    }
    spec.signal.rules = {
        "cross_a": SignalRuleDef(type="Crossover", params={"fast": "fast_a", "slow": "slow_a"}),
        "cross_b": SignalRuleDef(type="Crossover", params={"fast": "fast_b", "slow": "slow_b"}),
    }

    with pytest.raises(ValueError, match="Multiple Crossover"):
        compile_run(spec, out_dir=tmp_path)


def test_metrics_artifact_hash_excludes_run_id(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="metrics_hash", hypothesis="run metadata is not metrics fingerprint")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    first = tmp_path / "run_1"
    second = tmp_path / "run_2"
    first.mkdir()
    second.mkdir()
    _write_artifacts(spec, result, first, Engine())
    _write_artifacts(spec, result, second, Engine())

    first_hashes = json.loads((first / "artifact_hashes.json").read_text(encoding="utf-8"))
    second_hashes = json.loads((second / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert first_hashes["metrics.json"] == second_hashes["metrics.json"]


def test_environment_artifact_hash_excludes_run_timestamp(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="environment_hash", hypothesis="volatile timestamps are not core hashes")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    first = tmp_path / "run_1"
    second = tmp_path / "run_2"
    first.mkdir()
    second.mkdir()
    _write_artifacts(spec, result, first, Engine())
    _write_artifacts(spec, result, second, Engine())

    first_hashes = json.loads((first / "artifact_hashes.json").read_text(encoding="utf-8"))
    second_hashes = json.loads((second / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert first_hashes["environment.json"] == second_hashes["environment.json"]


def test_metrics_json_sanitizes_non_finite_values(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="finite_metrics", hypothesis="metrics json must be standard json")
    dates = pd.bdate_range("2024-01-01", periods=1, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "volume": [1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    metrics_text = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert "NaN" not in metrics_text
    assert "Infinity" not in metrics_text
    json.loads(metrics_text)


def test_oos_metrics_include_test_start_baseline() -> None:
    spec = StrategySpec.template(strategy_id="oos_baseline", hypothesis="oos metrics include first test-day move")
    spec.validation.train_period = ["2024-01-01", "2024-01-02"]
    spec.validation.test_period = ["2024-01-03", "2024-01-05"]
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("80000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 80000.0), (dates[2], 80000.0)],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["oos_total_return"] == pytest.approx(-0.2)
    assert metrics["oos_max_drawdown"] == pytest.approx(-0.2)


def test_oos_metrics_handle_zero_baseline() -> None:
    spec = StrategySpec.template(strategy_id="oos_zero", hypothesis="zero baseline cannot produce returns")
    spec.validation.train_period = ["2024-01-01", "2024-01-02"]
    spec.validation.test_period = ["2024-01-03", "2024-01-05"]
    dates = pd.bdate_range("2024-01-02", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("0")),
        trades=[],
        equity_curve=[(dates[0], 0.0), (dates[1], 0.0), (dates[2], 0.0)],
        mktdata={},
    )

    metrics = _build_metrics(spec, result, "run_1")

    assert metrics["oos_total_return"] is None
    assert metrics["oos_sharpe_ratio"] is None
    assert metrics["oos_max_drawdown"] is None


def test_reproducibility_audit_fails_when_artifact_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="tamper_test", hypothesis="artifact tampering fails audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "metrics.json").write_text('{"strategy_id": "tampered"}\n', encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "metrics_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_data_manifest_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="manifest_tamper", hypothesis="manifest tampering fails audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["price_adjustment"] = "raw"
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_environment_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="environment_tamper", hypothesis="environment tampering fails audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    env = json.loads((tmp_path / "environment.json").read_text(encoding="utf-8"))
    env["data_dir"] = str(tmp_path / "other_data")
    (tmp_path / "environment.json").write_text(json.dumps(env), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "environment_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_raw_spec_file_is_tampered(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="raw_spec_tamper", hypothesis="raw spec file tampering fails audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    (tmp_path / "strategy_spec.yaml").write_text(
        (tmp_path / "strategy_spec.yaml").read_text(encoding="utf-8") + "\nunknown_field: changed\n",
        encoding="utf-8",
    )

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "strategy_spec_file_hash" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_handles_corrupt_metrics_json(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="corrupt_metrics", hypothesis="corrupt metrics fail audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "metrics.json").write_text("{not-json", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "metrics_hash" and check["status"] == "fail" for check in audit["checks"])


def test_data_manifest_uses_test_period_start_when_train_period_absent(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="test_only", hypothesis="test-only specs record actual range")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-01", "2024-01-03"]
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["start"] == "2024-01-01"
    assert manifest["end"] == "2024-01-03"


def test_data_manifest_start_includes_min_start_date(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="warmup_start", hypothesis="manifest includes warmup data range")
    spec.data.min_start_date = "2023-12-01"
    dates = pd.bdate_range("2023-12-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )

    _write_artifacts(spec, result, tmp_path, Engine())

    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["start"] == "2023-12-01"


def test_reproducibility_audit_handles_corrupt_artifact_hashes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="corrupt_hashes", hypothesis="corrupt hash manifest fails cleanly")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "artifact_hashes.json").write_text("{not-json", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_handles_non_dict_artifact_hashes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="list_hashes", hypothesis="hash manifest type fails cleanly")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "artifact_hashes.json").write_text("[]", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_handles_bad_artifact_hash_schema_version(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_hash_schema", hypothesis="bad hash schema fails cleanly")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["schema_version"] = "bad"
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_on_empty_artifact_hashes(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="empty_hashes", hypothesis="empty hash manifest fails")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "artifact_hashes.json").write_text("{}\n", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_fails_when_artifact_hash_key_is_missing(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="missing_hash_key", hypothesis="hash manifest keys are required")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes.pop("metrics.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_rejects_hash_manifest_schema_downgrade(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="hash_downgrade", hypothesis="new artifacts cannot downgrade hash schema")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result.mktdata["SPY"].to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes.pop("schema_version")
    hashes.pop("environment.json")
    hashes.pop("strategy_spec.yaml")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "artifact_hashes" and check["status"] == "fail" for check in audit["checks"])


def test_reproducibility_audit_rejects_fingerprint_symbol_path_traversal(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="unsafe_fingerprint_symbol", hypothesis="fingerprint symbols are paths")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprints"]["../outside"] = manifest["data_fingerprints"].pop("SPY")
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert "../outside" in fingerprint_check["message"]


def test_reproducibility_audit_rejects_malformed_v1_data_fingerprints(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_fingerprint_schema", hypothesis="fingerprints must be an object")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprints"] = ["SPY"]
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert fingerprint_check["severity"] == "fatal"


def test_reproducibility_audit_rejects_non_object_fingerprint_entries(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_fingerprint_entry", hypothesis="fingerprint entries must be objects")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprints"]["SPY"] = "bad"
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert fingerprint_check["severity"] == "fatal"
    assert "fingerprint must be an object" in fingerprint_check["message"]


def test_reproducibility_audit_rejects_invalid_data_manifest_schema_version(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_manifest_schema", hypothesis="manifest schema version must parse")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": df},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_parquet(data_dir / "SPY.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = "bad"
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest" and check["severity"] == "fatal" for check in audit["checks"])


def test_reproducibility_audit_rejects_non_object_environment_json(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_env_type", hypothesis="environment json must be an object")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": pd.DataFrame(
            {"open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0], "volume": [1, 1, 1]},
            index=dates,
        )},
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "environment.json").write_text("[]", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "environment" and check["severity"] == "fatal" for check in audit["checks"])


def test_reproducibility_audit_rejects_non_object_data_manifest_json(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_manifest_type", hypothesis="manifest json must be an object")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": pd.DataFrame(
            {"open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0], "volume": [1, 1, 1]},
            index=dates,
        )},
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    (tmp_path / "data_manifest.json").write_text("[]", encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest" and check["severity"] == "fatal" for check in audit["checks"])


def test_reproducibility_audit_rejects_invalid_manifest_symbols(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="bad_manifest_symbols", hypothesis="manifest symbols must be a list")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": pd.DataFrame(
            {"open": [1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.0], "low": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0], "volume": [1, 1, 1]},
            index=dates,
        )},
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["symbols"] = 1
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    assert any(check["id"] == "data_manifest" and check["severity"] == "fatal" for check in audit["checks"])


def test_reproducibility_audit_requires_fingerprints_for_all_manifest_symbols(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="fingerprint_coverage", hypothesis="all symbols need source fingerprints")
    spec.universe.symbols = ["SPY", "QQQ"]
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1, 1, 1],
        },
        index=dates,
    )
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={"SPY": frame, "QQQ": frame},
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    frame.to_parquet(data_dir / "SPY.parquet")
    frame.to_parquet(data_dir / "QQQ.parquet")
    _write_artifacts(spec, result, tmp_path, Engine(), effective_data_dir=str(data_dir))
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest["data_fingerprints"].pop("QQQ")
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes["data_manifest.json"] = _hash_json_file(tmp_path / "data_manifest.json")
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "fail"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert "QQQ" in fingerprint_check["message"]


def test_reproducibility_audit_allows_legacy_artifacts_without_source_fingerprints(tmp_path) -> None:
    spec = StrategySpec.template(strategy_id="legacy_artifact", hypothesis="old artifacts use legacy audit")
    dates = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    result = RunResult(
        portfolio=Portfolio(cash=Decimal("100000")),
        trades=[],
        equity_curve=[(dates[0], 100000.0), (dates[1], 100001.0), (dates[2], 100003.0)],
        mktdata={
            "SPY": pd.DataFrame(
                {
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.0, 1.0, 1.0],
                    "low": [1.0, 1.0, 1.0],
                    "close": [1.0, 1.0, 1.0],
                    "volume": [1, 1, 1],
                },
                index=dates,
            )
        },
    )
    _write_artifacts(spec, result, tmp_path, Engine())
    manifest = json.loads((tmp_path / "data_manifest.json").read_text(encoding="utf-8"))
    manifest.pop("schema_version", None)
    manifest.pop("data_fingerprints", None)
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    current_hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashes = {
        "data_manifest.json": "sha256:" + hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()[:16],
        "equity_curve.csv": current_hashes["equity_curve.csv"],
        "trades.csv": current_hashes["trades.csv"],
        "metrics.json": current_hashes["metrics.json"],
    }
    (tmp_path / "artifact_hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    (tmp_path.parent / "run_digests.jsonl").write_text(
        json.dumps({"run_id": tmp_path.name, "artifact_hashes": _hash_json_file(tmp_path / "artifact_hashes.json")}) + "\n",
        encoding="utf-8",
    )

    audit = audit_reproducibility(tmp_path)

    assert audit["status"] == "pass"
    fingerprint_check = next(check for check in audit["checks"] if check["id"] == "data_fingerprint")
    assert fingerprint_check["status"] == "fail"
    assert fingerprint_check["severity"] == "warning"


def test_crossover_latch_can_be_reset_after_exit() -> None:
    spec = StrategySpec.template(strategy_id="cross_reset", hypothesis="crossover exits clear active entry state")
    spec.signal.indicators = {
        "fast": IndicatorDef(type="SMA", params={"period": 2}),
        "slow": IndicatorDef(type="SMA", params={"period": 3}),
    }
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"}),
    }

    optimizer = _build_optimizer(spec)
    entry_bar = pd.DataFrame({"cross": [True]})
    inactive_bar = pd.DataFrame({"cross": [False]})

    assert optimizer.optimize({"SPY": entry_bar}, {"SPY": entry_bar}) == {"SPY": 1.0}

    optimizer.reset_symbols(["SPY"])

    assert optimizer.optimize({"SPY": inactive_bar}, {"SPY": inactive_bar}) == {"CASH": 1.0}


def test_signal_filtered_equal_weight_ignores_negative_numeric_signals() -> None:
    spec = StrategySpec.template(strategy_id="negative_signal", hypothesis="negative signals should not go long")
    spec.signal.rules = {
        "threshold": SignalRuleDef(type="Threshold", params={"column": "score", "threshold": 0.0}),
    }

    optimizer = _build_optimizer(spec)
    signal_bar = pd.DataFrame({"threshold": [-1]})

    assert optimizer.optimize({"SPY": signal_bar}, {"SPY": signal_bar}) == {"CASH": 1.0}


def test_signal_filtered_equal_weight_keeps_latched_event_symbol_without_current_bar() -> None:
    spec = StrategySpec.template(strategy_id="sparse_latch", hypothesis="latched event holdings survive sparse calendars")
    spec.signal.rules = {
        "cross": SignalRuleDef(type="Crossover", params={"fast": "fast", "slow": "slow"}),
    }

    optimizer = _build_optimizer(spec)
    entry_bar = pd.DataFrame({"cross": [True]})

    assert optimizer.optimize({"SPY": entry_bar}, {"SPY": entry_bar}) == {"SPY": 1.0}
    optimizer.set_held_symbols(["SPY"])
    assert optimizer.optimize({}, {}) == {"SPY": 1.0}


def test_compile_strategy_rejects_unsupported_universe_type() -> None:
    spec = StrategySpec.template(strategy_id="unsupported_universe", hypothesis="unsupported universes fail clearly")
    spec.universe.type = "filter"

    with pytest.raises(ValueError, match="Unsupported universe.type 'filter'"):
        compile_strategy(spec)
