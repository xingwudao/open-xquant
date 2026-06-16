"""oxq CLI — Agentic Quant Research Kernel command-line interface."""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from oxq.spec.schema import StrategySpec
from oxq.spec.validator import validate as validate_spec


@click.group()
def main():
    """oxq — Agentic Quant Research Kernel CLI."""


@main.group()
def spec():
    """Manage strategy specs."""


@spec.command()
@click.argument("description")
@click.option("--out", "-o", default="strategy_spec.yaml", help="Output file path")
def init(description: str, out: str):
    """Initialize a new strategy spec from a natural language description.

    DESCRIPTION is a brief strategy idea in natural language.
    """
    strategy_id = description.lower().replace(" ", "_").replace("，", "_").replace(",", "_")[:50]
    template = StrategySpec.template(strategy_id=strategy_id, hypothesis=description)

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(template.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False), encoding="utf-8")

    click.echo(f"Spec template written to {output_path}")
    click.echo(f"Strategy ID: {strategy_id}")
    click.echo("Next: edit the file, then run `oxq spec validate`")


@spec.command()
@click.argument("spec_file", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def validate(spec_file: str, as_json: bool):
    """Validate a strategy spec file.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    try:
        parsed = StrategySpec.from_yaml(spec_file)
    except Exception as e:
        result = {
            "status": "fail",
            "errors": [{"severity": "fatal", "check": "parse_error", "message": str(e)}],
            "warnings": [],
            "spec_hash": "",
        }
        if as_json:
            import json

            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"FAIL: {e}")
        raise SystemExit(1)

    result = validate_spec(parsed)

    if as_json:
        import json

        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(f"Status: {result.status.upper()}")
        click.echo(f"Spec Hash: {result.spec_hash}")
        if result.errors:
            click.echo(f"\nErrors ({len(result.errors)}):")
            for e in result.errors:
                click.echo(f"  [{e['severity']}] {e['check']}: {e['message']}")
        if result.warnings:
            click.echo(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                click.echo(f"  [{w['severity']}] {w['check']}: {w['message']}")
        if result.status == "pass":
            click.echo("\nSpec is valid.")

    if result.status == "fail":
        raise SystemExit(1)


@main.group()
def backtest():
    """Run backtests from strategy specs."""


@backtest.command()
@click.argument("spec_file", type=click.Path(exists=True))
@click.option("--out", "-o", default="runs/auto", help="Output directory for run artifacts")
@click.option("--data-dir", default=None, help="Directory for market data files")
def run(spec_file: str, out: str, data_dir: str | None):
    """Run a backtest from a strategy spec file.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    from oxq.spec.compiler import compile_run

    spec = StrategySpec.from_yaml(spec_file)
    validation = validate_spec(spec)
    if validation.status == "fail":
        click.echo("Spec validation failed. Fix errors before running backtest:")
        for e in validation.errors:
            click.echo(f"  [{e['severity']}] {e['check']}: {e['message']}")
        raise SystemExit(1)
    if validation.warnings:
        click.echo("Warnings (continuing):")
        for w in validation.warnings:
            click.echo(f"  [{w['severity']}] {w['check']}: {w['message']}")

    click.echo(f"Running backtest for '{spec.strategy_id}'...")
    result, run_dir = compile_run(spec, data_dir=data_dir, out_dir=out)

    click.echo(f"\nRun complete. Artifacts written to {run_dir}/")
    click.echo(f"  Total Return: {result.total_return():.2%}")
    click.echo(f"  Sharpe Ratio: {result.sharpe_ratio():.2f}")
    click.echo(f"  Max Drawdown: {result.max_drawdown():.2%}")
    click.echo(f"  Trade Count:  {len(result.trades)}")


@main.group()
def strategy():
    """Manage compiled strategies."""


@strategy.command()
@click.argument("spec_file", type=click.Path(exists=True))
def compile(spec_file: str):
    """Compile a strategy spec into an executable strategy.

    SPEC_FILE is the path to a strategy_spec.yaml file.
    """
    from oxq.spec.compiler import compile_strategy

    spec = StrategySpec.from_yaml(spec_file)
    validation = validate_spec(spec)
    if validation.status == "fail":
        click.echo("Spec validation failed:")
        for e in validation.errors:
            click.echo(f"  [{e['severity']}] {e['check']}: {e['message']}")
        raise SystemExit(1)

    strategy_obj = compile_strategy(spec)
    click.echo(f"Strategy '{strategy_obj.name}' compiled successfully.")
    click.echo(f"  Universe:  {spec.universe.type} ({len(spec.universe.symbols)} symbols)")
    click.echo(f"  Signals:   {list(spec.signal.rules.keys())}")
    click.echo(f"  Portfolio: {spec.portfolio.type}")
    click.echo(f"  Hash:      {spec.compute_hash()}")


@main.group()
def audit():
    """Audit backtest runs for reproducibility and research bias."""


@audit.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def reproducibility(run_dir: str, as_json: bool):
    """Run reproducibility audit on a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    from oxq.audit import audit_reproducibility

    result = audit_reproducibility(run_dir)

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, indent=2))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        click.echo(f"Fatal: {result['fatal_count']}, Warnings: {result['warning_count']}")
        for c in result["checks"]:
            icon = "PASS" if c["status"] == "pass" else "FAIL"
            click.echo(f"  [{c['severity']}] {icon} {c['id']}: {c['message']}")

    if result["status"] == "fail":
        raise SystemExit(1)


@audit.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def research(run_dir: str, as_json: bool):
    """Run research bias audit on a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    from oxq.audit import audit_research

    result = audit_research(run_dir)

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, indent=2))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        click.echo(f"Fatal: {result['fatal_count']}, Warnings: {result['warning_count']}")
        for c in result["checks"]:
            icon = "PASS" if c["status"] == "pass" else "FAIL"
            click.echo(f"  [{c['severity']}] {icon} {c['id']}: {c['message']}")

    if result["status"] == "fail":
        raise SystemExit(1)


@main.group()
def report():
    """Generate research reports from backtest runs."""


@report.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--out", "-o", default=None, help="Output file path (default: run_dir/research_report.md)")
def write(run_dir: str, out: str | None):
    """Generate a research report from a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    from oxq.report import generate_report

    report_md = generate_report(run_dir)
    output_path = Path(out) if out else Path(run_dir) / "research_report.md"
    output_path.write_text(report_md, encoding="utf-8")
    click.echo(f"Report written to {output_path}")


@main.group()
def experiment():
    """Manage experiment registry."""


@experiment.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--registry", "-r", default="experiments.jsonl", help="Experiment registry file")
def add(run_dir: str, registry: str):
    """Add a backtest run to the experiment registry.

    RUN_DIR is the path to a run directory.
    """
    from oxq.observe.experiment_registry import add_experiment

    if not (Path(run_dir) / "metrics.json").exists():
        click.echo("Error: metrics.json not found in run directory")
        raise SystemExit(1)

    entry = add_experiment(run_dir, registry_path=registry)
    if "error" in entry:
        click.echo(f"Error: {entry['error']}")
        raise SystemExit(1)

    click.echo(f"Experiment added to {registry}")
    click.echo(f"  Experiment ID: {entry['experiment_id']}")
    click.echo(f"  Strategy:      {entry['strategy_id']}")


@main.group()
def robustness():
    """Run robustness tests on backtest runs."""


@robustness.command(name="run")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def run_robustness_cmd(run_dir: str, as_json: bool):
    """Run robustness tests on a backtest run directory.

    RUN_DIR is the path to a run directory (e.g. runs/20260616_153000_strategy_id/).
    """
    import oxq.robustness

    result = oxq.robustness.run_robustness(run_dir)

    if as_json:
        import json as _json

        click.echo(_json.dumps(result, indent=2, default=str))
    else:
        click.echo(f"Status: {result['status'].upper()}")
        click.echo(f"Baseline Sharpe: {result.get('baseline_sharpe', 0):.4f}")
        click.echo("")
        for t in result["tests"]:
            icon = "PASS" if t["status"] == "pass" else ("FAIL" if t["status"] == "fail" else "WARN")
            click.echo(f"  [{t['status'].upper()}] {icon} {t['name']}: {t.get('message', '')}")
            if "baseline_sharpe" in t:
                click.echo(f"         Baseline: {t['baseline_sharpe']:.4f} → Perturbed: {t['perturbed_sharpe']:.4f}")
