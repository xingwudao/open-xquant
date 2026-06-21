from __future__ import annotations

import json
from pathlib import Path

import pytest

from oxq.report.assets import (
    add_report_asset,
    add_report_assets,
    list_report_assets,
    manifest_path,
    safe_asset_id,
)


def test_safe_asset_id_accepts_simple_ids() -> None:
    assert safe_asset_id("equity_vs_benchmark") == "equity_vs_benchmark"
    assert safe_asset_id("drawdown-curve") == "drawdown-curve"


@pytest.mark.parametrize(
    "asset_id",
    ["", ".", "..", "../x", "a/b", "a\\b", "chart#1", "chart?1", "chart%2e", "chart 1", "chart)1", "chart&1"],
)
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


def test_add_report_asset_keeps_source_scripts_with_same_basename_distinct(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"png")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_script = first_dir / "plot.py"
    second_script = second_dir / "plot.py"
    first_script.write_text("print('first')\n", encoding="utf-8")
    second_script.write_text("print('second')\n", encoding="utf-8")

    first = add_report_asset(run_dir, figure, asset_id="first_chart", title="First", source_script=first_script)
    second = add_report_asset(run_dir, figure, asset_id="second_chart", title="Second", source_script=second_script)

    assert first.source.script == "scripts/plot.py"
    assert second.source.script == "scripts/second_chart_plot.py"
    assert (run_dir / "report_assets" / first.source.script).read_text(encoding="utf-8") == "print('first')\n"
    assert (run_dir / "report_assets" / second.source.script).read_text(encoding="utf-8") == "print('second')\n"


def test_add_report_asset_avoids_existing_asset_prefixed_script_collision(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"png")
    first_dir = tmp_path / "first"
    occupied_dir = tmp_path / "occupied"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    occupied_dir.mkdir()
    second_dir.mkdir()
    first_script = first_dir / "plot.py"
    occupied_script = occupied_dir / "second_chart_plot.py"
    second_script = second_dir / "plot.py"
    first_script.write_text("print('first')\n", encoding="utf-8")
    occupied_script.write_text("print('occupied')\n", encoding="utf-8")
    second_script.write_text("print('second')\n", encoding="utf-8")

    add_report_asset(run_dir, figure, asset_id="first_chart", title="First", source_script=first_script)
    occupied = add_report_asset(run_dir, figure, asset_id="occupied_chart", title="Occupied", source_script=occupied_script)
    second = add_report_asset(run_dir, figure, asset_id="second_chart", title="Second", source_script=second_script)

    assert occupied.source.script == "scripts/second_chart_plot.py"
    assert second.source.script != "scripts/second_chart_plot.py"
    assert (run_dir / "report_assets" / occupied.source.script).read_text(encoding="utf-8") == "print('occupied')\n"
    assert (run_dir / "report_assets" / second.source.script).read_text(encoding="utf-8") == "print('second')\n"


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


def test_add_report_asset_upserts_existing_id_after_in_place_regeneration(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source = tmp_path / "source.png"
    source.write_bytes(b"first")

    add_report_asset(run_dir, source, asset_id="same", title="First")
    regenerated = run_dir / "report_assets/figures/same.png"
    regenerated.write_bytes(b"regenerated")

    add_report_asset(run_dir, regenerated, asset_id="same", title="Regenerated")

    assets = list_report_assets(run_dir)
    assert len(assets) == 1
    assert assets[0].title == "Regenerated"
    assert assets[0].sha256 != "sha256:stale"
    assert regenerated.read_bytes() == b"regenerated"


def test_add_report_assets_upserts_multiple_in_place_regenerated_assets(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    equity_source = tmp_path / "equity.png"
    drawdown_source = tmp_path / "drawdown.png"
    equity_source.write_bytes(b"equity-first")
    drawdown_source.write_bytes(b"drawdown-first")

    add_report_asset(run_dir, equity_source, asset_id="equity", title="Equity")
    add_report_asset(run_dir, drawdown_source, asset_id="drawdown", title="Drawdown")
    (run_dir / "report_assets/figures/equity.png").write_bytes(b"equity-regenerated")
    (run_dir / "report_assets/figures/drawdown.png").write_bytes(b"drawdown-regenerated")

    assets = add_report_assets(
        run_dir,
        [
            {
                "id": "equity",
                "file_path": str(run_dir / "report_assets/figures/equity.png"),
                "title": "Equity Updated",
                "section": "results",
                "order": 10,
            },
            {
                "id": "drawdown",
                "file_path": str(run_dir / "report_assets/figures/drawdown.png"),
                "title": "Drawdown Updated",
                "section": "risk",
                "order": 20,
            },
        ],
    )

    assert [asset.id for asset in assets] == ["equity", "drawdown"]
    listed = list_report_assets(run_dir)
    assert [asset.id for asset in listed] == ["equity", "drawdown"]
    assert [asset.title for asset in listed] == ["Equity Updated", "Drawdown Updated"]
    assert (run_dir / "report_assets/figures/equity.png").read_bytes() == b"equity-regenerated"
    assert (run_dir / "report_assets/figures/drawdown.png").read_bytes() == b"drawdown-regenerated"


def test_add_report_assets_still_validates_assets_outside_batch(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    equity_source = tmp_path / "equity.png"
    stale_source = tmp_path / "stale.png"
    equity_source.write_bytes(b"equity")
    stale_source.write_bytes(b"stale")

    add_report_asset(run_dir, equity_source, asset_id="equity", title="Equity")
    add_report_asset(run_dir, stale_source, asset_id="stale", title="Stale")
    (run_dir / "report_assets/figures/stale.png").write_bytes(b"changed outside batch")
    (run_dir / "report_assets/figures/equity.png").write_bytes(b"equity-regenerated")

    with pytest.raises(ValueError, match="hash mismatch for report asset stale"):
        add_report_assets(
            run_dir,
            [
                {
                    "id": "equity",
                    "file_path": str(run_dir / "report_assets/figures/equity.png"),
                    "title": "Equity Updated",
                }
            ],
        )


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


def test_add_report_asset_accepts_source_script_already_in_report_assets_with_mixed_paths(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    figure = run_dir / "report_assets/figures/equity.png"
    script = run_dir / "report_assets/scripts/plot.py"
    figure.parent.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    figure.write_bytes(b"png")
    script.write_text("print('plot')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    asset = add_report_asset(Path("run"), figure.resolve(), asset_id="equity", title="策略净值", source_script=script.resolve())

    assert asset.source.script == "scripts/plot.py"
    assert script.read_text(encoding="utf-8") == "print('plot')\n"


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


@pytest.mark.parametrize(
    "manifest_path_value",
    [
        "../outside.png",
        "/tmp/outside.png",
        "scripts/plot.py",
        "figures/../outside.png",
        "figures/%2e%2e/outside.png",
        "figures/chart 1.png",
        "figures/chart)1.png",
    ],
)
def test_list_report_assets_rejects_unsafe_manifest_paths(tmp_path, manifest_path_value: str) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = manifest_path(run_dir)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "bad",
                        "kind": "figure",
                        "path": manifest_path_value,
                        "title": "Bad",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid report asset path"):
        list_report_assets(run_dir)


def test_list_report_assets_rejects_hash_mismatch(tmp_path) -> None:
    run_dir = tmp_path / "run"
    figure = run_dir / "report_assets/figures/equity.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b"changed")
    manifest_path(run_dir).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "equity",
                        "kind": "figure",
                        "path": "figures/equity.png",
                        "title": "Equity",
                        "sha256": "sha256:stale",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        list_report_assets(run_dir)
