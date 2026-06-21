# Report Assets And Chart Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build report asset registration, Chinese-first Markdown and HTML reports, and a `report-chart-builder` Agent skill that guides chart requirement discussion and Agent-written plotting scripts.

**Architecture:** Add a focused asset manager for `report_assets/manifest.json`, then introduce a `ReportBundle` assembly layer and separate Markdown/HTML renderers. Keep chart generation outside `oxq report write`; Agents or users create figures, then register them with `oxq report asset add`.

**Tech Stack:** Python 3.12+, Click CLI, dataclasses, JSON, pathlib, stdlib `html`, existing `pytest` test suite, existing open-xquant artifacts.

---

## File Structure

- Create `src/oxq/report/assets.py`
  - Own `ReportAsset` data, manifest read/write, safe IDs, copying files,
    hashing, and asset sorting.
- Create `src/oxq/report/bundle.py`
  - Build `ReportBundle` from a run directory.
  - Keep artifact reading and report decision separate from rendering.
- Create `src/oxq/report/i18n.py`
  - Store `zh` and `en` labels.
  - Provide a strict `messages(lang)` helper.
- Create `src/oxq/report/markdown.py`
  - Render `ReportBundle` to Markdown.
  - Embed registered figure assets and link attachments.
- Create `src/oxq/report/html.py`
  - Render `ReportBundle` to offline static notebook-like HTML.
- Modify `src/oxq/report/generator.py`
  - Keep `generate_report(run_dir, lang="zh") -> str`.
  - Re-export current decision and formatter helpers as needed for existing
    tests.
- Modify `src/oxq/report/__init__.py`
  - Export new public helpers.
- Modify `src/oxq/cli/main.py`
  - Add `oxq report asset add/list`.
  - Upgrade `oxq report write` with `--lang` and `--format`.
- Modify `src/oxq/tools/report.py`
  - Keep tool compatibility while returning Markdown and HTML outputs.
- Create `agent/skills/report-chart-builder.md`
  - Guide Agents through discussing chart needs, writing plotting scripts,
    registering assets, and generating reports.
- Modify docs and examples:
  - `docs/agent-guide.md`
  - `docs/human-guide.md`
  - `docs/architecture.md`
  - `examples/modules/05_report_and_experiment.py`
  - `agent/opencode/commands/quant-report.md`
  - `agent/opencode/agents/quant-reporter.md`
- Create tests:
  - `tests/report/test_assets.py`
  - `tests/report/test_renderers.py`
  - `tests/cli/test_report_assets.py`
  - `tests/agent/test_report_chart_builder_skill.py`

---

### Task 1: Asset Manager

**Files:**
- Create: `src/oxq/report/assets.py`
- Test: `tests/report/test_assets.py`

- [ ] **Step 1: Write asset manager tests**

Create `tests/report/test_assets.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/report/test_assets.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'oxq.report.assets'
```

- [ ] **Step 3: Implement `src/oxq/report/assets.py`**

Create `src/oxq/report/assets.py`:

```python
"""Report asset manifest management."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EMBEDDED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AssetSource:
    script: str | None = None
    input_artifacts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "AssetSource":
        if not isinstance(raw, dict):
            return cls()
        artifacts = raw.get("input_artifacts", [])
        return cls(
            script=raw.get("script") if isinstance(raw.get("script"), str) else None,
            input_artifacts=[str(item) for item in artifacts] if isinstance(artifacts, list) else [],
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.script:
            data["script"] = self.script
        if self.input_artifacts:
            data["input_artifacts"] = self.input_artifacts
        return data


@dataclass(frozen=True)
class ReportAsset:
    id: str
    kind: str
    path: str
    title: str
    caption: str = ""
    section: str = "results"
    order: int = 100
    mime_type: str = "application/octet-stream"
    sha256: str = ""
    source: AssetSource = field(default_factory=AssetSource)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ReportAsset":
        return cls(
            id=str(raw["id"]),
            kind=str(raw["kind"]),
            path=str(raw["path"]),
            title=str(raw["title"]),
            caption=str(raw.get("caption", "")),
            section=str(raw.get("section", "results")),
            order=int(raw.get("order", 100)),
            mime_type=str(raw.get("mime_type", "application/octet-stream")),
            sha256=str(raw.get("sha256", "")),
            source=AssetSource.from_dict(raw.get("source")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "path": self.path,
            "title": self.title,
            "caption": self.caption,
            "section": self.section,
            "order": self.order,
            "mime_type": self.mime_type,
            "sha256": self.sha256,
        }
        source = self.source.to_dict()
        if source:
            data["source"] = source
        return data


def report_assets_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / "report_assets"


def manifest_path(run_dir: str | Path) -> Path:
    return report_assets_dir(run_dir) / "manifest.json"


def safe_asset_id(asset_id: str) -> str:
    candidate = asset_id.strip()
    if not candidate or "/" in candidate or "\\" in candidate:
        raise ValueError(f"invalid asset id: {asset_id}")
    path = Path(candidate)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"invalid asset id: {asset_id}")
    return candidate


def list_report_assets(run_dir: str | Path) -> list[ReportAsset]:
    path = manifest_path(run_dir)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    assets = raw.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError(f"invalid report asset manifest: {path}")
    parsed = [ReportAsset.from_dict(item) for item in assets if isinstance(item, dict)]
    return sorted(parsed, key=lambda item: (item.section, item.order, item.id))


def add_report_asset(
    run_dir: str | Path,
    file_path: str | Path,
    *,
    asset_id: str,
    title: str,
    caption: str = "",
    section: str = "results",
    order: int = 100,
    source_script: str | Path | None = None,
    source_artifacts: list[str] | None = None,
) -> ReportAsset:
    run_path = Path(run_dir)
    if not run_path.exists():
        raise FileNotFoundError(f"run directory not found: {run_path}")
    source = Path(file_path)
    if not source.exists():
        raise FileNotFoundError(f"asset file not found: {source}")
    asset_name = safe_asset_id(asset_id)
    extension = source.suffix.lower()
    kind = "figure" if extension in EMBEDDED_IMAGE_EXTENSIONS else "attachment"
    subdir = "figures" if kind == "figure" else "attachments"
    assets_root = report_assets_dir(run_path)
    dest_dir = assets_root / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{asset_name}{extension}"
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)

    script_rel: str | None = None
    if source_script is not None:
        script_source = Path(source_script)
        if not script_source.exists():
            raise FileNotFoundError(f"source script not found: {script_source}")
        script_dir = assets_root / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_dest = script_dir / script_source.name
        if script_source.resolve() != script_dest.resolve():
            shutil.copy2(script_source, script_dest)
        script_rel = script_dest.relative_to(assets_root).as_posix()

    mime_type = mimetypes.guess_type(dest.name)[0] or "application/octet-stream"
    asset = ReportAsset(
        id=asset_name,
        kind=kind,
        path=dest.relative_to(assets_root).as_posix(),
        title=title,
        caption=caption,
        section=section,
        order=int(order),
        mime_type=mime_type,
        sha256=_sha256_file(dest),
        source=AssetSource(script=script_rel, input_artifacts=list(source_artifacts or [])),
    )
    _upsert_asset(run_path, asset)
    return asset


def _upsert_asset(run_dir: Path, asset: ReportAsset) -> None:
    assets_root = report_assets_dir(run_dir)
    assets_root.mkdir(parents=True, exist_ok=True)
    existing = [item for item in list_report_assets(run_dir) if item.id != asset.id]
    existing.append(asset)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "assets": [item.to_dict() for item in sorted(existing, key=lambda item: (item.section, item.order, item.id))],
    }
    manifest_path(run_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
```

- [ ] **Step 4: Run tests to verify asset manager passes**

Run:

```bash
uv run pytest tests/report/test_assets.py -v
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit asset manager**

Run:

```bash
git add src/oxq/report/assets.py tests/report/test_assets.py
git commit -m "add report asset manifest manager"
```

---

### Task 2: Report Asset CLI

**Files:**
- Modify: `src/oxq/cli/main.py`
- Test: `tests/cli/test_report_assets.py`

- [ ] **Step 1: Write CLI tests**

Create `tests/cli/test_report_assets.py`:

```python
from __future__ import annotations

import json

from click.testing import CliRunner

from oxq.cli.main import main


def test_report_asset_add_and_list(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    figure = tmp_path / "equity.png"
    figure.write_bytes(b"png")
    script = tmp_path / "plot_equity.py"
    script.write_text("print('plot')\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "report",
            "asset",
            "add",
            str(run_dir),
            str(figure),
            "--id",
            "equity_vs_benchmark",
            "--title",
            "策略净值与基准对比",
            "--caption",
            "由 equity_curve.csv 生成。",
            "--section",
            "results",
            "--order",
            "10",
            "--source-script",
            str(script),
            "--source-artifact",
            "equity_curve.csv",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "equity_vs_benchmark" in result.output
    assert (run_dir / "report_assets/figures/equity_vs_benchmark.png").exists()
    assert (run_dir / "report_assets/scripts/plot_equity.py").exists()

    manifest = json.loads((run_dir / "report_assets/manifest.json").read_text(encoding="utf-8"))
    assert manifest["assets"][0]["source"]["input_artifacts"] == ["equity_curve.csv"]

    listed = CliRunner().invoke(main, ["report", "asset", "list", str(run_dir)])
    assert listed.exit_code == 0, listed.output
    assert "equity_vs_benchmark" in listed.output
    assert "策略净值与基准对比" in listed.output


def test_report_asset_list_empty_state(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    result = CliRunner().invoke(main, ["report", "asset", "list", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert "No report assets registered" in result.output
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
uv run pytest tests/cli/test_report_assets.py -v
```

Expected:

```text
Error: No such command 'asset'
```

- [ ] **Step 3: Add nested `report asset` commands**

Modify `src/oxq/cli/main.py` near the existing `report` group:

```python
@report.group()
def asset():
    """Manage report assets for experiment reports."""


@asset.command(name="add")
@click.argument("run_dir", type=click.Path(exists=True))
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--id", "asset_id", required=True, help="Stable asset id")
@click.option("--title", required=True, help="Asset title")
@click.option("--caption", default="", help="Asset caption")
@click.option("--section", default="results", help="Report section")
@click.option("--order", default=100, type=int, help="Ordering within the section")
@click.option("--source-script", default=None, type=click.Path(exists=True), help="Plotting script path")
@click.option("--source-artifact", "source_artifacts", multiple=True, help="Input artifact used by the asset")
def asset_add(
    run_dir: str,
    file_path: str,
    asset_id: str,
    title: str,
    caption: str,
    section: str,
    order: int,
    source_script: str | None,
    source_artifacts: tuple[str, ...],
):
    """Register a report asset and update report_assets/manifest.json."""
    from oxq.report.assets import add_report_asset

    registered = add_report_asset(
        run_dir,
        file_path,
        asset_id=asset_id,
        title=title,
        caption=caption,
        section=section,
        order=order,
        source_script=source_script,
        source_artifacts=list(source_artifacts),
    )
    click.echo(f"Asset registered: {registered.id}")
    click.echo(f"  kind:   {registered.kind}")
    click.echo(f"  path:   report_assets/{registered.path}")
    click.echo(f"  sha256: {registered.sha256}")


@asset.command(name="list")
@click.argument("run_dir", type=click.Path(exists=True))
def asset_list(run_dir: str):
    """List registered report assets."""
    from oxq.report.assets import list_report_assets

    assets = list_report_assets(run_dir)
    if not assets:
        click.echo("No report assets registered.")
        return
    for item in assets:
        click.echo(f"{item.id} [{item.kind}] {item.title}")
        click.echo(f"  section/order: {item.section}/{item.order}")
        click.echo(f"  path: report_assets/{item.path}")
        click.echo(f"  sha256: {item.sha256}")
```

- [ ] **Step 4: Run CLI tests**

Run:

```bash
uv run pytest tests/cli/test_report_assets.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit report asset CLI**

Run:

```bash
git add src/oxq/cli/main.py tests/cli/test_report_assets.py
git commit -m "add report asset cli"
```

---

### Task 3: ReportBundle And Language Catalog

**Files:**
- Create: `src/oxq/report/bundle.py`
- Create: `src/oxq/report/i18n.py`
- Modify: `src/oxq/report/__init__.py`
- Test: `tests/report/test_renderers.py`

- [ ] **Step 1: Write bundle and language tests**

Create `tests/report/test_renderers.py` with the first tests:

```python
from __future__ import annotations

import json

import yaml

from oxq.report.bundle import build_report_bundle
from oxq.report.i18n import messages
from oxq.spec.schema import StrategySpec


def test_messages_default_to_chinese() -> None:
    zh = messages("zh")
    assert zh["report_title"] == "实验报告"
    assert zh["chart_assets"] == "图表资产"


def test_messages_support_english() -> None:
    en = messages("en")
    assert en["report_title"] == "Research Report"
    assert en["chart_assets"] == "Chart Assets"


def test_build_report_bundle_reads_assets(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    assets_root = run_dir / "report_assets"
    (assets_root / "figures").mkdir(parents=True)
    (assets_root / "figures/equity.png").write_bytes(b"png")
    (assets_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "equity",
                        "kind": "figure",
                        "path": "figures/equity.png",
                        "title": "策略净值",
                        "caption": "测试图表",
                        "section": "results",
                        "order": 10,
                        "mime_type": "image/png",
                        "sha256": "sha256:abc",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bundle = build_report_bundle(run_dir)

    assert bundle.strategy_id == "bundle_test"
    assert bundle.assets[0].id == "equity"
    assert bundle.metrics["run_id"] == "bundle-run"


def _write_report_run(tmp_path):
    spec = StrategySpec.template(strategy_id="bundle_test", hypothesis="bundle hypothesis")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "bundle-run",
                "trade_count": 2,
                "max_drawdown": -0.05,
                "oos_max_drawdown": -0.05,
                "oos_sharpe_ratio": 1.2,
                "total_return": 0.1,
                "annualized_return": 0.08,
                "annualized_volatility": 0.12,
                "sharpe_ratio": 1.1,
            }
        ),
        encoding="utf-8",
    )
    return run_dir
```

- [ ] **Step 2: Run bundle tests to verify they fail**

Run:

```bash
uv run pytest tests/report/test_renderers.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'oxq.report.bundle'
```

- [ ] **Step 3: Implement language catalog**

Create `src/oxq/report/i18n.py`:

```python
"""Report language labels."""

from __future__ import annotations


CATALOGS: dict[str, dict[str, str]] = {
    "zh": {
        "report_title": "实验报告",
        "generated": "生成时间",
        "run_id": "运行 ID",
        "executive_decision": "执行结论",
        "hypothesis": "研究假设",
        "strategy_summary": "策略配置摘要",
        "data_execution": "数据与执行假设",
        "metrics": "回测指标",
        "benchmark": "基准对比",
        "reproducibility_audit": "可复现性审计",
        "research_audit": "研究偏差审计",
        "validation_classification": "校验分类",
        "robustness": "稳健性测试",
        "failure_modes": "失败模式",
        "chart_assets": "图表资产",
        "attachments": "附件资产",
        "asset_appendix": "资产清单",
        "next_actions": "下一步",
        "no_chart_assets": "本实验未登记图表资产。",
        "not_specified": "未指定",
    },
    "en": {
        "report_title": "Research Report",
        "generated": "Generated",
        "run_id": "Run ID",
        "executive_decision": "Executive Decision",
        "hypothesis": "Hypothesis",
        "strategy_summary": "Strategy Spec Summary",
        "data_execution": "Data and Execution Assumptions",
        "metrics": "Backtest Metrics",
        "benchmark": "Benchmark Comparison",
        "reproducibility_audit": "Reproducibility Audit",
        "research_audit": "Research Bias Audit",
        "validation_classification": "Validation Classification",
        "robustness": "Robustness Tests",
        "failure_modes": "Failure Modes",
        "chart_assets": "Chart Assets",
        "attachments": "Attachment Assets",
        "asset_appendix": "Asset Appendix",
        "next_actions": "Next Actions",
        "no_chart_assets": "No chart assets registered for this experiment.",
        "not_specified": "not specified",
    },
}


def messages(lang: str = "zh") -> dict[str, str]:
    try:
        return CATALOGS[lang]
    except KeyError as exc:
        raise ValueError(f"unsupported report language: {lang}") from exc
```

- [ ] **Step 4: Implement bundle assembly**

Create `src/oxq/report/bundle.py`:

```python
"""Structured report artifact assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from oxq.audit.reproducibility import audit_reproducibility
from oxq.audit.research_bias import audit_research
from oxq.report.assets import ReportAsset, list_report_assets
from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate


@dataclass(frozen=True)
class ReportBundle:
    run_dir: Path
    spec: StrategySpec
    spec_dict: dict[str, Any]
    metrics: dict[str, Any]
    execution_assumptions: dict[str, Any] | None
    repro_audit: dict[str, Any]
    bias_audit: dict[str, Any]
    validation_result: dict[str, Any]
    robustness_result: dict[str, Any] | None
    assets: list[ReportAsset]
    decision: str

    @property
    def strategy_id(self) -> str:
        return self.spec.strategy_id or "unknown"

    @property
    def hypothesis(self) -> str:
        return self.spec.research.hypothesis or ""

    @property
    def run_id(self) -> str:
        return str(self.metrics.get("run_id", self.run_dir.name))


def build_report_bundle(run_dir: str | Path) -> ReportBundle:
    from oxq.report.generator import _determine_decision, _load_execution_assumptions, _load_verified_robustness_result

    run_path = Path(run_dir)
    spec = StrategySpec.from_yaml(str(run_path / "strategy_spec.yaml"))
    spec_dict = yaml.safe_load((run_path / "strategy_spec.yaml").read_text(encoding="utf-8")) or {}
    metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
    execution_assumptions = _load_execution_assumptions(run_path)
    repro_audit = audit_reproducibility(run_path)
    robustness_result = _load_verified_robustness_result(run_path, repro_audit)
    bias_audit = audit_research(run_path)
    validation_result = validate(spec).to_dict()
    decision = _determine_decision(bias_audit, spec_dict, metrics, repro_audit, robustness_result)
    return ReportBundle(
        run_dir=run_path,
        spec=spec,
        spec_dict=spec_dict,
        metrics=metrics,
        execution_assumptions=execution_assumptions,
        repro_audit=repro_audit,
        bias_audit=bias_audit,
        validation_result=validation_result,
        robustness_result=robustness_result,
        assets=list_report_assets(run_path),
        decision=decision,
    )
```

- [ ] **Step 5: Update public exports**

Modify `src/oxq/report/__init__.py`:

```python
from oxq.report.assets import ReportAsset, add_report_asset, list_report_assets
from oxq.report.bundle import ReportBundle, build_report_bundle
from oxq.report.generator import generate_report

__all__ = [
    "ReportAsset",
    "ReportBundle",
    "add_report_asset",
    "build_report_bundle",
    "generate_report",
    "list_report_assets",
]
```

- [ ] **Step 6: Run bundle tests**

Run:

```bash
uv run pytest tests/report/test_renderers.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 7: Commit bundle and i18n**

Run:

```bash
git add src/oxq/report/__init__.py src/oxq/report/bundle.py src/oxq/report/i18n.py tests/report/test_renderers.py
git commit -m "add report bundle and language catalog"
```

---

### Task 4: Markdown Renderer

**Files:**
- Create: `src/oxq/report/markdown.py`
- Modify: `src/oxq/report/generator.py`
- Test: `tests/report/test_generator.py`
- Test: `tests/report/test_renderers.py`

- [ ] **Step 1: Add Markdown asset rendering tests**

Append to `tests/report/test_renderers.py`:

```python
from oxq.report.markdown import render_markdown_report


def test_markdown_renderer_embeds_registered_figures(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    assets_root = run_dir / "report_assets"
    (assets_root / "figures").mkdir(parents=True)
    (assets_root / "figures/equity.png").write_bytes(b"png")
    (assets_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "equity",
                        "kind": "figure",
                        "path": "figures/equity.png",
                        "title": "策略净值",
                        "caption": "由 equity_curve.csv 生成。",
                        "section": "results",
                        "order": 10,
                        "mime_type": "image/png",
                        "sha256": "sha256:abc",
                        "source": {"script": "scripts/plot.py", "input_artifacts": ["equity_curve.csv"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle = build_report_bundle(run_dir)

    markdown = render_markdown_report(bundle, lang="zh")

    assert "# 实验报告: bundle_test" in markdown
    assert "## 12. 图表资产" in markdown
    assert "![策略净值](report_assets/figures/equity.png)" in markdown
    assert "sha256:abc" in markdown
    assert "scripts/plot.py" in markdown


def test_markdown_renderer_reports_empty_chart_assets(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    bundle = build_report_bundle(run_dir)

    markdown = render_markdown_report(bundle, lang="zh")

    assert "本实验未登记图表资产。" in markdown
```

- [ ] **Step 2: Run renderer tests to verify they fail**

Run:

```bash
uv run pytest tests/report/test_renderers.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'oxq.report.markdown'
```

- [ ] **Step 3: Implement Markdown renderer**

Create `src/oxq/report/markdown.py`.

Start with this code and move the existing Markdown section logic from
`src/oxq/report/generator.py` into `render_markdown_report`:

```python
"""Markdown report renderer."""

from __future__ import annotations

from datetime import UTC, datetime

from oxq.report.assets import ReportAsset
from oxq.report.bundle import ReportBundle
from oxq.report.i18n import messages
from oxq.spec.execution import derive_execution_semantics
from oxq.spec.schema import StrategySpec


def render_markdown_report(bundle: ReportBundle, lang: str = "zh") -> str:
    msg = messages(lang)
    spec = bundle.spec
    metrics = bundle.metrics
    lines: list[str] = []
    lines.append(f"# {msg['report_title']}: {bundle.strategy_id}")
    lines.append("")
    lines.append(f"**{msg['generated']}**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**{msg['run_id']}**: {bundle.run_id}")
    lines.append("")
    lines.extend(_core_sections(bundle, msg))
    lines.extend(_asset_sections(bundle, msg))
    lines.extend(_next_actions(bundle, msg))
    lines.append("")
    return "\n".join(lines)
```

Also define focused helpers in the same file:

```python
def _asset_sections(bundle: ReportBundle, msg: dict[str, str]) -> list[str]:
    figures = [asset for asset in bundle.assets if asset.kind == "figure"]
    attachments = [asset for asset in bundle.assets if asset.kind != "figure"]
    lines = ["## 12. " + msg["chart_assets"], ""]
    if not figures:
        lines.extend([msg["no_chart_assets"], ""])
    for index, asset in enumerate(figures, start=1):
        lines.extend(_figure_markdown(index, asset))
    if attachments:
        lines.extend(["## 13. " + msg["attachments"], ""])
        for asset in attachments:
            lines.append(f"- [{asset.title}](report_assets/{asset.path})")
        lines.append("")
    lines.extend(["## 14. " + msg["asset_appendix"], ""])
    if not bundle.assets:
        lines.extend([msg["no_chart_assets"], ""])
    for asset in bundle.assets:
        lines.append(f"- **{asset.id}** ({asset.kind}): `report_assets/{asset.path}`")
        lines.append(f"  - sha256: `{asset.sha256}`")
        if asset.source.script:
            lines.append(f"  - script: `report_assets/{asset.source.script}`")
        if asset.source.input_artifacts:
            lines.append(f"  - inputs: {', '.join(f'`{item}`' for item in asset.source.input_artifacts)}")
    lines.append("")
    return lines


def _figure_markdown(index: int, asset: ReportAsset) -> list[str]:
    lines = [
        f"### {asset.title}",
        "",
        f"![{asset.title}](report_assets/{asset.path})",
        "",
    ]
    if asset.caption:
        lines.extend([f"图 {index}. {asset.caption}", ""])
    return lines
```

Move or import these existing helper functions from `generator.py` so existing
formatting remains stable:

- `_decision_explanation`
- `_effective_fill_price_mode`
- `_format_assumption_value`
- `_format_execution_assumption_lines`
- `_format_float`
- `_format_is_oos_metric_lines`
- `_format_metric_assumption_lines`
- `_format_money`
- `_format_percent`
- `_format_robustness_result_lines`
- `_format_validation_classification_lines`
- `_has_is_oos_metrics`

Keep public helper imports in `generator.py` intact for existing tests.

- [ ] **Step 4: Keep `generate_report` backward compatible**

Modify `src/oxq/report/generator.py` so `generate_report` delegates:

```python
def generate_report(run_dir: str | Path, lang: str = "zh") -> str:
    """Generate a Markdown research report from a backtest run directory."""
    from oxq.report.bundle import build_report_bundle
    from oxq.report.markdown import render_markdown_report

    return render_markdown_report(build_report_bundle(run_dir), lang=lang)
```

Leave `_determine_decision` and existing formatter helpers importable from
`oxq.report.generator` until tests and external users migrate.

- [ ] **Step 5: Run report tests**

Run:

```bash
uv run pytest tests/report/test_generator.py tests/report/test_renderers.py -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 6: Commit Markdown renderer**

Run:

```bash
git add src/oxq/report/generator.py src/oxq/report/markdown.py tests/report/test_generator.py tests/report/test_renderers.py
git commit -m "add markdown report asset rendering"
```

---

### Task 5: HTML Renderer And Report Write Formats

**Files:**
- Create: `src/oxq/report/html.py`
- Modify: `src/oxq/cli/main.py`
- Modify: `src/oxq/report/__init__.py`
- Test: `tests/report/test_renderers.py`
- Test: `tests/cli/test_report_assets.py`

- [ ] **Step 1: Add HTML renderer tests**

Append to `tests/report/test_renderers.py`:

```python
from oxq.report.html import render_html_report


def test_html_renderer_embeds_registered_figures(tmp_path) -> None:
    run_dir = _write_report_run(tmp_path)
    assets_root = run_dir / "report_assets"
    (assets_root / "figures").mkdir(parents=True)
    (assets_root / "figures/equity.png").write_bytes(b"png")
    (assets_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "assets": [
                    {
                        "id": "equity",
                        "kind": "figure",
                        "path": "figures/equity.png",
                        "title": "策略净值",
                        "caption": "由 equity_curve.csv 生成。",
                        "section": "results",
                        "order": 10,
                        "mime_type": "image/png",
                        "sha256": "sha256:abc",
                        "source": {"script": "scripts/plot.py", "input_artifacts": ["equity_curve.csv"]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    bundle = build_report_bundle(run_dir)

    html = render_html_report(bundle, lang="zh")

    assert "<!doctype html>" in html
    assert "<h1>实验报告: bundle_test</h1>" in html
    assert '<img src="report_assets/figures/equity.png" alt="策略净值">' in html
    assert "由 equity_curve.csv 生成。" in html
    assert "sha256:abc" in html
```

- [ ] **Step 2: Add CLI format tests**

Append to `tests/cli/test_report_assets.py`:

```python
import yaml

from oxq.spec.schema import StrategySpec


def test_report_write_default_writes_markdown_and_html(tmp_path) -> None:
    run_dir = _write_cli_report_run(tmp_path)

    result = CliRunner().invoke(main, ["report", "write", str(run_dir)])

    assert result.exit_code == 0, result.output
    assert (run_dir / "research_report.md").exists()
    assert (run_dir / "research_report.html").exists()
    assert "research_report.md" in result.output
    assert "research_report.html" in result.output


def test_report_write_html_only(tmp_path) -> None:
    run_dir = _write_cli_report_run(tmp_path)

    result = CliRunner().invoke(main, ["report", "write", str(run_dir), "--format", "html"])

    assert result.exit_code == 0, result.output
    assert not (run_dir / "research_report.md").exists()
    assert (run_dir / "research_report.html").exists()


def _write_cli_report_run(tmp_path):
    spec = StrategySpec.template(strategy_id="cli_report", hypothesis="cli report")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "cli-run",
                "trade_count": 2,
                "max_drawdown": -0.05,
                "oos_max_drawdown": -0.05,
                "oos_sharpe_ratio": 1.2,
                "total_return": 0.1,
                "annualized_return": 0.08,
                "annualized_volatility": 0.12,
                "sharpe_ratio": 1.1,
            }
        ),
        encoding="utf-8",
    )
    return run_dir
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/report/test_renderers.py tests/cli/test_report_assets.py -v
```

Expected:

```text
ModuleNotFoundError: No module named 'oxq.report.html'
```

- [ ] **Step 4: Implement HTML renderer**

Create `src/oxq/report/html.py`:

```python
"""Static HTML report renderer."""

from __future__ import annotations

from html import escape

from oxq.report.bundle import ReportBundle
from oxq.report.i18n import messages
from oxq.report.markdown import render_markdown_report


def render_html_report(bundle: ReportBundle, lang: str = "zh") -> str:
    msg = messages(lang)
    figures = [asset for asset in bundle.assets if asset.kind == "figure"]
    markdown_summary = escape(render_markdown_report(bundle, lang=lang))
    figure_html = "\n".join(_figure_html(asset) for asset in figures)
    if not figure_html:
        figure_html = f"<p>{escape(msg['no_chart_assets'])}</p>"
    return f"""<!doctype html>
<html lang="{escape(lang)}">
<head>
  <meta charset="utf-8">
  <title>{escape(msg['report_title'])}: {escape(bundle.strategy_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; line-height: 1.55; }}
    main {{ max-width: 1080px; margin: 0 auto; }}
    section {{ margin: 28px 0; }}
    pre {{ background: #f6f8fa; padding: 16px; overflow-x: auto; }}
    figure {{ margin: 24px 0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    figcaption {{ color: #555; margin-top: 8px; }}
    code {{ background: #f6f8fa; padding: 1px 4px; }}
  </style>
</head>
<body>
<main>
  <h1>{escape(msg['report_title'])}: {escape(bundle.strategy_id)}</h1>
  <section>
    <h2>{escape(msg['chart_assets'])}</h2>
    {figure_html}
  </section>
  <section>
    <h2>{escape(msg['asset_appendix'])}</h2>
    {_asset_appendix_html(bundle)}
  </section>
  <section>
    <h2>Markdown Source</h2>
    <pre>{markdown_summary}</pre>
  </section>
</main>
</body>
</html>
"""


def _figure_html(asset) -> str:
    caption = escape(asset.caption)
    metadata = _asset_metadata_html(asset)
    return (
        "<figure>"
        f'<img src="report_assets/{escape(asset.path)}" alt="{escape(asset.title)}">'
        f"<figcaption><strong>{escape(asset.title)}</strong>"
        f"{'<br>' + caption if caption else ''}"
        f"{metadata}</figcaption>"
        "</figure>"
    )


def _asset_metadata_html(asset) -> str:
    chunks = [f"<br><code>{escape(asset.sha256)}</code>"]
    if asset.source.script:
        chunks.append(f"<br>script: <code>report_assets/{escape(asset.source.script)}</code>")
    if asset.source.input_artifacts:
        inputs = ", ".join(escape(item) for item in asset.source.input_artifacts)
        chunks.append(f"<br>inputs: <code>{inputs}</code>")
    return "".join(chunks)


def _asset_appendix_html(bundle: ReportBundle) -> str:
    if not bundle.assets:
        return "<p>No assets.</p>"
    items = []
    for asset in bundle.assets:
        items.append(
            "<li>"
            f"<strong>{escape(asset.id)}</strong> "
            f"({escape(asset.kind)}): "
            f"<code>report_assets/{escape(asset.path)}</code> "
            f"<code>{escape(asset.sha256)}</code>"
            "</li>"
        )
    return "<ul>" + "".join(items) + "</ul>"
```

- [ ] **Step 5: Add report writing helper and CLI options**

Modify `src/oxq/report/__init__.py`:

```python
from oxq.report.html import render_html_report
from oxq.report.markdown import render_markdown_report
```

Modify `src/oxq/cli/main.py` report write command:

```python
@report.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--out", "-o", default=None, help="Output file path for markdown-only compatibility")
@click.option("--lang", default="zh", type=click.Choice(["zh", "en"]), help="Report language")
@click.option("--format", "output_format", default="all", type=click.Choice(["all", "markdown", "html"]), help="Report format")
def write(run_dir: str, out: str | None, lang: str, output_format: str):
    """Generate research reports from a backtest run directory."""
    from oxq.report.bundle import build_report_bundle
    from oxq.report.html import render_html_report
    from oxq.report.markdown import render_markdown_report

    run_path = Path(run_dir)
    bundle = build_report_bundle(run_path)
    written: list[Path] = []
    if output_format in {"all", "markdown"}:
        markdown_path = Path(out) if out else run_path / "research_report.md"
        markdown_path.write_text(render_markdown_report(bundle, lang=lang), encoding="utf-8")
        written.append(markdown_path)
    if output_format in {"all", "html"}:
        html_path = run_path / "research_report.html"
        html_path.write_text(render_html_report(bundle, lang=lang), encoding="utf-8")
        written.append(html_path)
    for path in written:
        click.echo(f"Report written to {path}")
```

- [ ] **Step 6: Run renderer and CLI tests**

Run:

```bash
uv run pytest tests/report/test_renderers.py tests/cli/test_report_assets.py -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 7: Commit HTML renderer and format CLI**

Run:

```bash
git add src/oxq/report/html.py src/oxq/report/__init__.py src/oxq/cli/main.py tests/report/test_renderers.py tests/cli/test_report_assets.py
git commit -m "add html report rendering"
```

---

### Task 6: MCP Tool Compatibility

**Files:**
- Modify: `src/oxq/tools/report.py`
- Test: `tests/tools/test_report.py`

- [ ] **Step 1: Write tool tests**

Create `tests/tools/test_report.py`:

```python
from __future__ import annotations

import json

import yaml

from oxq.spec.schema import StrategySpec
from oxq.tools.report import report_write


def test_report_write_tool_returns_markdown_and_html_outputs(tmp_path) -> None:
    run_dir = _write_tool_report_run(tmp_path)

    result = report_write(str(run_dir))

    assert result["status"] == "ok"
    assert result["markdown_output"].endswith("research_report.md")
    assert result["html_output"].endswith("research_report.html")
    assert (run_dir / "research_report.md").exists()
    assert (run_dir / "research_report.html").exists()


def _write_tool_report_run(tmp_path):
    spec = StrategySpec.template(strategy_id="tool_report", hypothesis="tool report")
    spec.validation.train_period = []
    spec.validation.test_period = ["2024-01-02", "2024-01-03"]
    spec.validation.required_oos = False
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "tool-run",
                "trade_count": 2,
                "max_drawdown": -0.05,
                "oos_max_drawdown": -0.05,
                "oos_sharpe_ratio": 1.2,
                "total_return": 0.1,
                "annualized_return": 0.08,
                "annualized_volatility": 0.12,
                "sharpe_ratio": 1.1,
            }
        ),
        encoding="utf-8",
    )
    return run_dir
```

- [ ] **Step 2: Run tool test to verify it fails**

Run:

```bash
uv run pytest tests/tools/test_report.py -v
```

Expected:

```text
KeyError: 'markdown_output'
```

- [ ] **Step 3: Update report tool**

Modify `src/oxq/tools/report.py`:

```python
def report_write(run_dir: str, out: str | None = None, lang: str = "zh", output_format: str = "all") -> dict[str, Any]:
    """Generate Markdown and/or HTML reports from a backtest run."""
    from oxq.report.bundle import build_report_bundle
    from oxq.report.html import render_html_report
    from oxq.report.markdown import render_markdown_report

    run_path = Path(run_dir)
    bundle = build_report_bundle(run_path)
    markdown_path: Path | None = None
    html_path: Path | None = None
    if output_format in {"all", "markdown"}:
        markdown_path = Path(out) if out else run_path / "research_report.md"
        markdown_path.write_text(render_markdown_report(bundle, lang=lang), encoding="utf-8")
    if output_format in {"all", "html"}:
        html_path = run_path / "research_report.html"
        html_path.write_text(render_html_report(bundle, lang=lang), encoding="utf-8")

    return {
        "status": "ok",
        "markdown_output": str(markdown_path) if markdown_path else None,
        "html_output": str(html_path) if html_path else None,
        "strategy_id": bundle.strategy_id,
        "decision": bundle.decision,
    }
```

Update the registry description to mention Markdown, HTML, `lang`, and asset
manifest rendering.

- [ ] **Step 4: Run tool test**

Run:

```bash
uv run pytest tests/tools/test_report.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit tool compatibility**

Run:

```bash
git add src/oxq/tools/report.py tests/tools/test_report.py
git commit -m "update report tool outputs"
```

---

### Task 7: Report Chart Builder Skill

**Files:**
- Create: `agent/skills/report-chart-builder.md`
- Test: `tests/agent/test_report_chart_builder_skill.py`

- [ ] **Step 1: Write skill validation test**

Create `tests/agent/test_report_chart_builder_skill.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml


def test_report_chart_builder_skill_frontmatter_and_required_workflow() -> None:
    path = Path("agent/skills/report-chart-builder.md")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    end = text.index("\n---", 4)
    frontmatter = yaml.safe_load(text[4:end])

    assert frontmatter["name"] == "report-chart-builder"
    assert "chart assets" in frontmatter["description"]
    assert "plotting Python code" in frontmatter["description"]

    required_phrases = [
        "Confirm the run directory",
        "recommend chart options",
        "ask for confirmation",
        "report_assets/scripts/",
        "report_assets/figures/",
        "oxq report asset add",
        "oxq report write",
        "Do not use charts to override audit failures",
    ]
    for phrase in required_phrases:
        assert phrase in text
```

- [ ] **Step 2: Run skill test to verify it fails**

Run:

```bash
uv run pytest tests/agent/test_report_chart_builder_skill.py -v
```

Expected:

```text
FileNotFoundError: agent/skills/report-chart-builder.md
```

- [ ] **Step 3: Create skill**

Create `agent/skills/report-chart-builder.md`:

```markdown
---
name: report-chart-builder
description: >-
  Design and generate chart assets for open-xquant experiment reports; use when
  users ask to add charts to a report, visualize backtest artifacts for a
  report, decide what figures should appear in a notebook-like experiment
  report, or have an Agent write plotting Python code for report assets.
---

# Report Chart Builder

You help turn experiment artifacts into report chart assets. The report system
does not guess or draw charts by itself. You clarify chart requirements, write
plotting code when requested, register generated files, and regenerate the
Markdown and HTML reports.

## Workflow

1. Confirm the run directory.
2. Inspect standard artifacts:
   - `metrics.json`
   - `equity_curve.csv`
   - `benchmark_curve.csv` when present
   - `trades.csv`
   - `target_weights.csv`
   - `robustness.json` when present
   - `report_assets/manifest.json` when present
3. If the user requests a specific chart, verify required artifacts exist.
4. If the user does not know which charts to use, recommend chart options from
   available artifacts and ask for confirmation before generating anything.
5. Write plotting Python code only after chart requirements are clear.
6. Save plotting scripts under `report_assets/scripts/`.
7. Save generated images under `report_assets/figures/`.
8. Register every figure with `oxq report asset add`.
9. Generate reports with `oxq report write RUN_DIR --lang zh --format all`.
10. Report Markdown, HTML, manifest, figure, and script paths to the user.

## Common Recommendations

- `equity_curve.csv` plus `benchmark_curve.csv`: strategy equity versus
  benchmark.
- `equity_curve.csv`: drawdown curve.
- `trades.csv`: trade distribution.
- `target_weights.csv`: target weight changes.
- `robustness.json`: robustness summary visualization.

## Asset Registration Pattern

```bash
oxq report asset add runs/<run_id>/ \
  runs/<run_id>/report_assets/figures/equity_vs_benchmark.png \
  --id equity_vs_benchmark \
  --title "策略净值与基准对比" \
  --caption "由 equity_curve.csv 和 benchmark_curve.csv 生成。" \
  --section results \
  --order 10 \
  --source-script runs/<run_id>/report_assets/scripts/plot_equity_vs_benchmark.py \
  --source-artifact equity_curve.csv \
  --source-artifact benchmark_curve.csv
```

## Red Lines

- Do not invent charts without user confirmation unless the user explicitly
  asks you to recommend charts.
- Do not use charts to override audit failures.
- Do not hide missing source artifacts by plotting different data.
- Do not leave generated figures outside the run directory.
- Do not edit backtest artifacts.
- Do not describe a visual check as proof of profitability.
```

- [ ] **Step 4: Run skill test**

Run:

```bash
uv run pytest tests/agent/test_report_chart_builder_skill.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit skill**

Run:

```bash
git add agent/skills/report-chart-builder.md tests/agent/test_report_chart_builder_skill.py
git commit -m "add report chart builder skill"
```

---

### Task 8: Documentation And Example Updates

**Files:**
- Modify: `docs/agent-guide.md`
- Modify: `docs/human-guide.md`
- Modify: `docs/architecture.md`
- Modify: `examples/modules/05_report_and_experiment.py`
- Modify: `agent/opencode/commands/quant-report.md`
- Modify: `agent/opencode/agents/quant-reporter.md`

- [ ] **Step 1: Update docs with exact commands**

Add this command set to the report sections of the docs:

```bash
oxq report asset add runs/<run_id>/ /path/to/chart.png \
  --id equity_vs_benchmark \
  --title "策略净值与基准对比" \
  --caption "由 equity_curve.csv 和 benchmark_curve.csv 生成。" \
  --section results \
  --order 10 \
  --source-script runs/<run_id>/report_assets/scripts/plot_equity_vs_benchmark.py \
  --source-artifact equity_curve.csv \
  --source-artifact benchmark_curve.csv

oxq report asset list runs/<run_id>/
oxq report write runs/<run_id>/ --lang zh --format all
```

State these rules:

- Default report language is Chinese.
- Default outputs are `research_report.md` and `research_report.html`.
- `report write` does not create charts.
- Agent-created plotting scripts belong in `report_assets/scripts/`.
- Registered figures belong in `report_assets/figures/`.
- Report charts must be registered through `oxq report asset add`.

- [ ] **Step 2: Update example module**

Modify `examples/modules/05_report_and_experiment.py` to print:

```python
print(f"Markdown report: {run_dir / 'research_report.md'}")
print(f"HTML report:     {run_dir / 'research_report.html'}")
print(f"Asset manifest:  {run_dir / 'report_assets/manifest.json'}")
```

Update CLI equivalent block:

```python
print(f"""
{'='*60}
CLI equivalents:
  oxq report asset add {run_dir}/ /path/to/chart.png --id chart_id --title "图表标题"
  oxq report asset list {run_dir}/
  oxq report write {run_dir}/ --lang zh --format all
  oxq experiment add {run_dir}/ --registry {registry_path}
{'='*60}
""")
```

- [ ] **Step 3: Run doc grep checks**

Run:

```bash
rg -n "report asset add|report-chart-builder|research_report.html|--lang zh|--format all" docs examples agent/opencode agent/skills
```

Expected:

```text
matches in docs, example module, opencode command/agent, and report-chart-builder skill
```

- [ ] **Step 4: Commit docs**

Run:

```bash
git add docs/agent-guide.md docs/human-guide.md docs/architecture.md examples/modules/05_report_and_experiment.py agent/opencode/commands/quant-report.md agent/opencode/agents/quant-reporter.md
git commit -m "document report asset workflow"
```

---

### Task 9: Final Verification

**Files:**
- No new files unless a prior task uncovered a defect.

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/report tests/cli/test_report_assets.py tests/tools/test_report.py tests/agent/test_report_chart_builder_skill.py -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 2: Run existing report-adjacent tests**

Run:

```bash
uv run pytest tests/cli/test_main.py tests/tools/test_optional_dependency_isolation.py tests/report/test_generator.py -v
```

Expected:

```text
all tests passed
```

- [ ] **Step 3: Check CLI help**

Run:

```bash
uv run oxq report --help
uv run oxq report asset --help
uv run oxq report write --help
```

Expected:

```text
report help lists asset and write
asset help lists add and list
write help lists --lang and --format
```

- [ ] **Step 4: Run whitespace check**

Run:

```bash
git diff --check
```

Expected:

```text
no output
```

- [ ] **Step 5: Commit any verification fixes**

If Step 1 through Step 4 required fixes, commit them:

```bash
git add <fixed-files>
git commit -m "fix report asset verification issues"
```

If no fixes were needed, do not create a commit.

---

## Self-Review

- Spec coverage:
  - Asset manifest is covered by Tasks 1 and 2.
  - Markdown and HTML outputs are covered by Tasks 4 and 5.
  - Chinese default and English support are covered by Tasks 3 and 5.
  - Agent chart workflow is covered by Task 7.
  - Docs and examples are covered by Task 8.
  - Verification is covered by Task 9.
- Red-flag scan:
  - No task uses unresolved markers or vague deferred-work language.
- Type consistency:
  - Asset APIs use `ReportAsset`, `AssetSource`, `add_report_asset`,
    `list_report_assets`, and `build_report_bundle` consistently.
  - CLI uses `--format` mapped to `output_format` to avoid shadowing the
    Python built-in in function parameters.
