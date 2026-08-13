from __future__ import annotations

from collections.abc import ItemsView, Iterator, Mapping
from types import MappingProxyType

import pandas as pd
import pytest

from oxq.operators.errors import DuplicateKeyError, InvalidPanelError, MissingColumnError
from oxq.operators.panel import QuantPanelAdapter, validate_serialized_quant_panel
from oxq.operators.types import OperatorContext


def _serialized_daily_panel(**row_values: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "context": {
            "timezone": "Asia/Shanghai",
            "calendar": "XSHG",
            "frequency": "1d",
            "timestamp_semantics": "session_date",
            "currency": "CNY",
            "price_adjustment": "forward_adjusted",
            "data_version": "fixture-v1",
            "source": "fake",
        },
        "rows": [{"date": "2026-01-05", "code": "000001.SZ", **row_values}],
    }


def _deep_json_object(depth: int) -> dict[str, object]:
    root: dict[str, object] = {}
    current = root
    for _ in range(depth):
        child: dict[str, object] = {}
        current["nested"] = child
        current = child
    return root


class _VisitBudget:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.visits = 0

    def record(self) -> None:
        self.visits += 1
        if self.visits > self.limit:
            raise AssertionError("shared JSON DAG traversal exceeded its visit budget")


class _BudgetedMapping(Mapping[str, object]):
    def __init__(self, values: dict[str, object], budget: _VisitBudget) -> None:
        self._values = values
        self._budget = budget

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def items(self) -> ItemsView[str, object]:
        self._budget.record()
        return self._values.items()


def _compact_shared_json_dag(depth: int, budget: _VisitBudget) -> Mapping[str, object]:
    node: Mapping[str, object] = _BudgetedMapping({"leaf": 1}, budget)
    for _ in range(depth):
        node = _BudgetedMapping({"left": node, "right": node}, budget)
    return node


def test_daily_frames_round_trip_without_mutating_input(daily_context, daily_symbol_frames) -> None:
    snapshots = {code: frame.copy(deep=True) for code, frame in daily_symbol_frames.items()}

    panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)

    assert list(panel.columns) == ["date", "code", "close", "volume"]
    assert panel[["date", "code"]].values.tolist() == [
        [pd.Timestamp("2026-01-05"), "000001.SZ"],
        [pd.Timestamp("2026-01-05"), "600000.SH"],
        [pd.Timestamp("2026-01-06"), "000001.SZ"],
        [pd.Timestamp("2026-01-06"), "600000.SH"],
        [pd.Timestamp("2026-01-07"), "000001.SZ"],
        [pd.Timestamp("2026-01-07"), "600000.SH"],
    ]
    assert panel["date"].dt.tz is None
    restored = QuantPanelAdapter.to_symbol_frames(panel, daily_context)
    for code, original in snapshots.items():
        pd.testing.assert_frame_equal(daily_symbol_frames[code], original)
        pd.testing.assert_frame_equal(restored[code], original, check_freq=False)


def test_to_panel_recursively_copies_object_cells(daily_context) -> None:
    frames = {
        "600000.SH": pd.DataFrame(
            {
                "close": [20.0],
                "metadata": [{"tags": ["bank"]}],
            },
            index=pd.DatetimeIndex(["2026-01-06"], tz="Asia/Shanghai"),
        ),
        "000001.SZ": pd.DataFrame(
            {
                "close": [10.0],
                "metadata": [[{"tags": ["finance"]}]],
            },
            index=pd.DatetimeIndex(["2026-01-05"], tz="Asia/Shanghai"),
        ),
    }

    panel = QuantPanelAdapter.to_panel(frames, daily_context)

    assert panel[["date", "code"]].values.tolist() == [
        [pd.Timestamp("2026-01-05"), "000001.SZ"],
        [pd.Timestamp("2026-01-06"), "600000.SH"],
    ]
    pd.testing.assert_series_equal(panel[["close", "metadata"]].dtypes, frames["000001.SZ"].dtypes)

    panel.loc[0, "metadata"][0]["tags"].append("mutated")
    panel.loc[1, "metadata"]["tags"].append("mutated")

    assert frames["000001.SZ"].loc[:, "metadata"].iloc[0] == [{"tags": ["finance"]}]
    assert frames["600000.SH"].loc[:, "metadata"].iloc[0] == {"tags": ["bank"]}


def test_to_symbol_frames_recursively_copies_object_cells(daily_context) -> None:
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-06")],
            "code": pd.Series(["000001.SZ", "600000.SH"], dtype="string"),
            "close": [10.0, 20.0],
            "metadata": [[{"tags": ["finance"]}], {"tags": ["bank"]}],
        }
    )

    frames = QuantPanelAdapter.to_symbol_frames(panel, daily_context)

    frames["000001.SZ"].loc[:, "metadata"].iloc[0][0]["tags"].append("mutated")
    frames["600000.SH"].loc[:, "metadata"].iloc[0]["tags"].append("mutated")

    assert panel.loc[0, "metadata"] == [{"tags": ["finance"]}]
    assert panel.loc[1, "metadata"] == {"tags": ["bank"]}


def test_intraday_panel_is_timezone_aware_and_canonicalized_to_utc() -> None:
    context = OperatorContext(
        timezone="Asia/Shanghai",
        calendar="XSHG",
        frequency="1min",
        timestamp_semantics="bar_close",
        currency="CNY",
        price_adjustment="raw",
        data_version="fixture-v1",
        source="fake",
        evaluation_time="intraday_t",
    )
    frames = {
        "000001.SZ": pd.DataFrame(
            {"close": [10.0, 10.1]},
            index=pd.DatetimeIndex(["2026-01-05 09:31", "2026-01-05 09:32"], tz="Asia/Shanghai"),
        )
    }

    panel = QuantPanelAdapter.to_panel(frames, context)

    assert str(panel["date"].dt.tz) == "UTC"
    assert panel.loc[0, "date"] == pd.Timestamp("2026-01-05 01:31", tz="UTC")
    restored = QuantPanelAdapter.to_symbol_frames(panel, context)
    assert str(restored["000001.SZ"].index.tz) == "Asia/Shanghai"
    assert restored["000001.SZ"].index[0] == frames["000001.SZ"].index[0]


def test_intraday_rejects_naive_or_mixed_timezone_indexes() -> None:
    context = OperatorContext(
        timezone="Asia/Shanghai",
        calendar="XSHG",
        frequency="1min",
        timestamp_semantics="bar_close",
        currency="CNY",
        price_adjustment="raw",
        data_version="fixture-v1",
        source="fake",
        evaluation_time="intraday_t",
    )
    aware = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-01-05 09:31"], tz="Asia/Shanghai"))
    naive = pd.DataFrame({"close": [2.0]}, index=pd.DatetimeIndex(["2026-01-05 09:31"]))

    with pytest.raises(InvalidPanelError, match="timezone-aware"):
        QuantPanelAdapter.to_panel({"aware": aware, "naive": naive}, context)


def test_empty_intraday_panel_keeps_timezone_aware_date_dtype() -> None:
    context = OperatorContext(
        timezone="Asia/Shanghai",
        calendar="XSHG",
        frequency="1min",
        timestamp_semantics="bar_close",
        currency="CNY",
        price_adjustment="raw",
        data_version="fixture-v1",
        source="fake",
        evaluation_time="intraday_t",
    )

    panel = QuantPanelAdapter.to_panel({}, context)

    assert isinstance(panel["date"].dtype, pd.DatetimeTZDtype)
    assert str(panel["date"].dt.tz) == "UTC"
    QuantPanelAdapter.validate_panel(panel, context)


def test_panel_rejects_duplicate_keys_and_missing_identifiers(daily_context) -> None:
    duplicate = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-05")],
            "code": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 11.0],
        }
    )
    with pytest.raises(DuplicateKeyError, match="duplicate"):
        QuantPanelAdapter.validate_panel(duplicate, daily_context)
    with pytest.raises(MissingColumnError, match="code"):
        QuantPanelAdapter.validate_panel(duplicate.drop(columns="code"), daily_context)


@pytest.mark.parametrize(
    "columns",
    [
        [0, "code", "close"],
        ["date", "", "close"],
        ["date", "code", 0],
        ["date", "code", ""],
    ],
)
def test_panel_rejects_non_string_or_empty_column_labels(
    daily_context: OperatorContext,
    columns: list[object],
) -> None:
    panel = pd.DataFrame(
        [[pd.Timestamp("2026-01-05"), "000001.SZ", 10.0]],
        columns=columns,
    )

    with pytest.raises(InvalidPanelError, match="columns must be non-empty strings"):
        QuantPanelAdapter.validate_panel(panel, daily_context)


def test_validate_output_inherits_column_label_validation(daily_context: OperatorContext) -> None:
    input_panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-05")],
            "code": ["000001.SZ"],
            "close": [10.0],
        }
    )
    output_panel = input_panel[["date", "code"]].copy()
    output_panel[0] = 1.0

    with pytest.raises(InvalidPanelError, match="columns must be non-empty strings"):
        QuantPanelAdapter.validate_output(
            input_panel,
            output_panel,
            daily_context,
            alignment="preserve_input_order",
        )


def test_symbol_frames_reject_reserved_quant_panel_columns(daily_context) -> None:
    frame = pd.DataFrame(
        {"date": ["shadow"], "close": [10.0]},
        index=pd.DatetimeIndex(["2026-01-05"], tz="Asia/Shanghai"),
    )
    with pytest.raises(InvalidPanelError, match="reserved"):
        QuantPanelAdapter.to_panel({"000001.SZ": frame}, daily_context)


def test_panel_rejects_non_session_daily_timestamps(daily_context) -> None:
    panel = pd.DataFrame({"date": [pd.Timestamp("2026-01-05 12:00")], "code": ["000001.SZ"], "close": [10.0]})
    with pytest.raises(InvalidPanelError, match="normalized session dates"):
        QuantPanelAdapter.validate_panel(panel, daily_context)


def test_canonical_order_validation_compares_keys_not_index_labels(daily_context) -> None:
    panel = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-06"), pd.Timestamp("2026-01-05")],
            "code": ["000001.SZ", "000001.SZ"],
            "close": [11.0, 10.0],
        },
        index=[0, 0],
    )

    with pytest.raises(InvalidPanelError, match="canonical"):
        QuantPanelAdapter.validate_panel(panel, daily_context, require_canonical_order=True)


def test_validate_output_enforces_declared_alignment(daily_context, daily_symbol_frames) -> None:
    input_panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    output = input_panel[["date", "code"]].copy()
    output["sma_2"] = [None, None, 10.5, 19.0, 11.5, 19.5]

    QuantPanelAdapter.validate_output(input_panel, output, daily_context, alignment="canonical_order")
    with pytest.raises(InvalidPanelError, match="canonical_order"):
        QuantPanelAdapter.validate_output(
            input_panel,
            output.iloc[::-1].reset_index(drop=True),
            daily_context,
            alignment="canonical_order",
        )
    with pytest.raises(InvalidPanelError, match="same keys"):
        QuantPanelAdapter.validate_output(
            input_panel,
            output.iloc[:-1],
            daily_context,
            alignment="preserve_input_order",
        )


@pytest.mark.parametrize("alignment", ["preserve-input-order", "unknown"])
def test_validate_output_rejects_unsupported_alignment(
    daily_context: OperatorContext,
    daily_symbol_frames: dict[str, pd.DataFrame],
    alignment: str,
) -> None:
    input_panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    output_panel = input_panel[["date", "code"]].copy()

    with pytest.raises(InvalidPanelError, match="unsupported output alignment"):
        QuantPanelAdapter.validate_output(
            input_panel,
            output_panel,
            daily_context,
            alignment=alignment,  # type: ignore[arg-type]
        )


def test_explicit_keyed_output_allows_unique_subset(daily_context, daily_symbol_frames) -> None:
    input_panel = QuantPanelAdapter.to_panel(daily_symbol_frames, daily_context)
    output = input_panel.loc[input_panel["code"] == "000001.SZ", ["date", "code"]].copy()
    output["signal"] = 1.0

    QuantPanelAdapter.validate_output(
        input_panel,
        output,
        daily_context,
        alignment="explicit_keyed_output",
    )


def test_serialized_intraday_panel_normalizes_lowercase_utc_designators_for_duplicate_keys() -> None:
    payload = {
        "schema_version": 1,
        "context": {
            "timezone": "UTC",
            "calendar": "XNYS",
            "frequency": "1min",
            "timestamp_semantics": "bar_close",
            "currency": "USD",
            "price_adjustment": "raw",
            "data_version": "fixture-v1",
            "source": "fake",
        },
        "rows": [
            {"date": "2026-01-01t00:00:00z", "code": "AAPL", "close": 10.0},
            {"date": "2025-12-31T19:00:00-05:00", "code": "AAPL", "close": 11.0},
        ],
    }

    with pytest.raises(DuplicateKeyError, match="duplicate"):
        validate_serialized_quant_panel(payload)


def test_serialized_panel_accepts_nested_json_values() -> None:
    shared_statistics = {"count": 2, "weight": 1.25}
    payload = _serialized_daily_panel(
        metadata={
            "active": True,
            "tags": ["bank", None],
            "statistics": shared_statistics,
            "statistics_copy": shared_statistics,
        }
    )

    validate_serialized_quant_panel(payload)


def test_serialized_panel_visits_shared_containers_once_per_phase() -> None:
    budget = _VisitBudget(limit=4)
    shared = _BudgetedMapping({"count": 2}, budget)
    payload = _serialized_daily_panel(metadata={"primary": shared, "copy": shared})

    validate_serialized_quant_panel(payload)

    assert budget.visits == 2


def test_serialized_panel_handles_compact_shared_dag_with_bounded_work() -> None:
    depth = 48
    budget = _VisitBudget(limit=2 * (depth + 1))
    payload = _serialized_daily_panel(metadata=_compact_shared_json_dag(depth, budget))

    validate_serialized_quant_panel(payload)

    assert budget.visits == 2 * (depth + 1)


def test_serialized_panel_rechecks_depth_when_reusing_completed_container() -> None:
    shared: object = {"nested": {"leaf": 1}}
    deep_reference = shared
    for _ in range(59):
        deep_reference = {"nested": deep_reference}
    payload = _serialized_daily_panel(metadata={"shallow": shared, "deep": deep_reference})

    with pytest.raises(InvalidPanelError, match="nesting depth"):
        validate_serialized_quant_panel(payload)


def test_serialized_panel_accepts_nested_read_only_mappings() -> None:
    payload = MappingProxyType(
        {
            "schema_version": 1,
            "context": MappingProxyType(
                {
                    "timezone": "Asia/Shanghai",
                    "calendar": "XSHG",
                    "frequency": "1d",
                    "timestamp_semantics": "session_date",
                    "currency": "CNY",
                    "price_adjustment": "forward_adjusted",
                    "data_version": "fixture-v1",
                    "source": "fake",
                }
            ),
            "rows": [
                MappingProxyType(
                    {
                        "date": "2026-01-05",
                        "code": "000001.SZ",
                        "metadata": MappingProxyType({"statistics": MappingProxyType({"count": 2})}),
                    }
                )
            ],
        }
    )

    validate_serialized_quant_panel(payload)


def test_serialized_panel_rejects_context_timezone_outside_zoneinfo() -> None:
    payload = _serialized_daily_panel()
    payload["context"]["timezone"] = "Mars/Base"  # type: ignore[index]

    with pytest.raises(InvalidPanelError, match=r"timezone.*Mars/Base.*zoneinfo"):
        validate_serialized_quant_panel(payload)


@pytest.mark.parametrize(
    "value",
    [b"binary", ("tuple",), {"set"}, object()],
    ids=["bytes", "tuple", "set", "object"],
)
def test_serialized_panel_rejects_nested_non_json_values(value: object) -> None:
    payload = _serialized_daily_panel(metadata={"nested": [value]})

    with pytest.raises(InvalidPanelError, match=r"rows\[0\]\.metadata\.nested\[0\].*JSON") as exc_info:
        validate_serialized_quant_panel(payload)

    assert exc_info.value.code == "invalid_panel"


def test_serialized_panel_rejects_nested_non_string_object_keys() -> None:
    payload = _serialized_daily_panel(metadata={1: "numeric", "1": "string"})

    with pytest.raises(InvalidPanelError, match=r"rows\[0\]\.metadata\[1\].*string") as exc_info:
        validate_serialized_quant_panel(payload)

    assert exc_info.value.code == "invalid_panel"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_serialized_panel_rejects_nonfinite_numbers(value: float) -> None:
    payload = _serialized_daily_panel(value=value)

    with pytest.raises(InvalidPanelError, match=r"rows\[0\]\.value.*finite") as exc_info:
        validate_serialized_quant_panel(payload)

    assert exc_info.value.code == "invalid_panel"


@pytest.mark.parametrize("container_type", [list, dict], ids=["list", "object"])
def test_serialized_panel_rejects_cyclic_trees(container_type: type[list] | type[dict]) -> None:
    cyclic: list[object] | dict[str, object] = container_type()
    if isinstance(cyclic, list):
        cyclic.append(cyclic)
    else:
        cyclic["self"] = cyclic
    payload = _serialized_daily_panel(metadata=cyclic)

    with pytest.raises(InvalidPanelError, match=r"rows\[0\]\.metadata.*cyclic") as exc_info:
        validate_serialized_quant_panel(payload)

    assert exc_info.value.code == "invalid_panel"


@pytest.mark.parametrize("location", ["unknown", "metadata"])
def test_serialized_panel_rejects_excessively_deep_json_trees(location: str) -> None:
    nested = _deep_json_object(1_000)
    payload = _serialized_daily_panel(**({"metadata": nested} if location == "metadata" else {}))
    if location == "unknown":
        payload["unknown"] = nested

    with pytest.raises(InvalidPanelError, match="nesting depth") as exc_info:
        validate_serialized_quant_panel(payload)

    assert exc_info.value.code == "invalid_panel"
