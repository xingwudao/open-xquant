"""Module example: Report generation and experiment registry.

Demonstrates generating research_report.md, research_report.html,
and registering the experiment in the JSONL registry.

Run: uv run python examples/modules/05_report_and_experiment.py
"""

from pathlib import Path

from oxq.observe.experiment_registry import add_experiment, list_experiments
from oxq.report import write_report_files

# Find the latest run directory from 03_backtest
RUNS_DIR = Path("/tmp/oxq_examples/03_backtest")
run_dirs = sorted(
    [p for p in RUNS_DIR.glob("*") if "cost_x" not in p.name],
    key=lambda p: p.name, reverse=True,
)
if not run_dirs:
    print("Run 03_backtest_and_artifacts.py first.")
    raise SystemExit(1)

run_dir = run_dirs[0]
print(f"Generating report for: {run_dir.name}\n")

# ---------------------------------------------------------------------------
# 1. Generate Research Reports
# ---------------------------------------------------------------------------

outputs = write_report_files(run_dir, lang="zh", output_format="all")
report_path = outputs.markdown
html_path = outputs.html
if report_path is None or html_path is None:
    raise RuntimeError("expected both Markdown and HTML reports")
report_md = report_path.read_text(encoding="utf-8")

# Print the key sections
print("=" * 50)
print("RESEARCH REPORT SUMMARY")
print("=" * 50)
for line in report_md.split("\n")[:30]:
    if line.startswith("#"):
        print(f"\n{line}")
    elif line.startswith("|") or line.startswith("-"):
        print(line)

# Print the decision (first bold line before the legend)
for line in report_md.split("\n"):
    if line.startswith("**") and line.endswith("**") and len(line) < 40:
        print(f"\n>>> Decision: {line.strip('*').strip()}")
        break

# ---------------------------------------------------------------------------
# 2. Register experiment
# ---------------------------------------------------------------------------

registry_path = Path("/tmp/oxq_examples/experiments.jsonl")
entry = add_experiment(run_dir, registry_path=registry_path)
print("\nExperiment registered:")
print(f"  ID:       {entry['experiment_id']}")
print(f"  Strategy: {entry['strategy_id']}")
print(f"  Registry: {registry_path}")

# ---------------------------------------------------------------------------
# 3. List all experiments
# ---------------------------------------------------------------------------

experiments = list_experiments(registry_path)
print(f"\nAll experiments ({len(experiments)} total):")
for exp in experiments:
    metrics = exp.get("metrics", {})
    sharpe = metrics.get("sharpe_ratio", 0)
    print(f"  {exp['experiment_id']} | {exp['strategy_id']:25s} | Sharpe={sharpe:.2f} | {exp['audit_status']}")

print(f"\nMarkdown report: {report_path}")
print(f"HTML report: {html_path}")
print(f"  tail -40 {report_path}")

# ---------------------------------------------------------------------------
# CLI equivalents
# ---------------------------------------------------------------------------
print(f"""
{'='*60}
Agent workflow:
  /quant-report {run_dir}/
  oxq experiment add {run_dir}/ --registry {registry_path}
{'='*60}
""")
