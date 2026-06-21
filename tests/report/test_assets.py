from __future__ import annotations

import json

import pytest

from oxq.report.assets import (
    add_report_asset,
    list_report_assets,
    manifest_path,
    safe_asset_id,
)


def test_safe_asset_id_accepts_simple_ids() -> None:
    assert safe_asset_id("equity_vs_benchmark") == "equity_vs_benchmark"
    assert safe_asset_id("drawdown-curve") == "drawdown-curve"


@pytest.mark.parametrize("asset_id", ["", ".", "..", "../x", "a/b", "a\\b"])
def test_safe_asset_id_rejects_path_like_ids(asset_id: str) -> None:
    with pytest.raises(ValueError, match="invalid asset id"):
        safe_asset_id(asset_id)


def test_add_report_asset_copies_figure_and_writes_manifest(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "equity.png"
    source.write_bytes(b"fake png bytes")
    script = tmp_path / "plot_equity.py"
    script.write_text("print('plot')\n", encoding="utf-8")

    asset = add_report_asset(
        run_dir,
        source,
        asset_id="equity_vs_benchmark",
        title="策略净值与基准对比",
        caption="由 equity_curve.csv 和 benchmark_curve.csv 生成。",
        section="results",
        order=10,
        source_script=script,
        source_artifacts=["equity_curve.csv", "benchmark_curve.csv"],
    )

    assert asset.id == "equity_vs_benchmark"
    assert asset.kind == "figure"
    assert asset.path == "figures/equity_vs_benchmark.png"
    assert asset.source.script == "scripts/plot_equity.py"
    assert asset.source.input_artifacts == ["equity_curve.csv", "benchmark_curve.csv"]
    assert asset.sha256.startswith("sha256:")
    assert (run_dir / "report_assets/figures/equity_vs_benchmark.png").read_bytes() == b"fake png bytes"
    assert (run_dir / "report_assets/scripts/plot_equity.py").read_text(encoding="utf-8") == "print('plot')\n"

    manifest = json.loads(manifest_path(run_dir).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["assets"][0]["id"] == "equity_vs_benchmark"


def test_add_report_asset_upserts_existing_id(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    add_report_asset(run_dir, first, asset_id="same", title="First")
    add_report_asset(run_dir, second, asset_id="same", title="Second", order=2)

    assets = list_report_assets(run_dir)
    assert len(assets) == 1
    assert assets[0].title == "Second"
    assert assets[0].path == "figures/same.png"
    assert (run_dir / "report_assets/figures/same.png").read_bytes() == b"second"


def test_add_report_asset_registers_attachment(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "notes.pdf"
    source.write_bytes(b"pdf")

    asset = add_report_asset(run_dir, source, asset_id="notes", title="补充说明")

    assert asset.kind == "attachment"
    assert asset.path == "attachments/notes.pdf"
    assert (run_dir / "report_assets/attachments/notes.pdf").read_bytes() == b"pdf"


def test_add_report_asset_accepts_file_already_in_report_assets(tmp_path) -> None:
    run_dir = tmp_path / "run"
    figure = run_dir / "report_assets/figures/equity.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"png")

    asset = add_report_asset(run_dir, figure, asset_id="equity", title="策略净值")

    assert asset.kind == "figure"
    assert asset.path == "figures/equity.png"
    assert figure.read_bytes() == b"png"


def test_list_report_assets_sorts_by_section_order_and_id(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    add_report_asset(run_dir, b, asset_id="b", title="B", section="risk", order=20)
    add_report_asset(run_dir, a, asset_id="a", title="A", section="results", order=10)

    assert [asset.id for asset in list_report_assets(run_dir)] == ["a", "b"]


def test_list_report_assets_returns_empty_without_manifest(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert list_report_assets(run_dir) == []
