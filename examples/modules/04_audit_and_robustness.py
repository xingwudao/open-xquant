"""Module example: Audit system and robustness tests.

Demonstrates running reproducibility audit, research bias audit,
and robustness tests on a completed backtest run.

Run: uv run python examples/modules/04_audit_and_robustness.py
"""

import json
import math
from pathlib import Path

from oxq.audit import audit_reproducibility, audit_research
from oxq.robustness import run_robustness


def _format_metric(value):
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{value:.4f}"
    return "N/A"

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
print(f"Auditing run: {run_dir.name}\n")

# ---------------------------------------------------------------------------
# 1. Reproducibility Audit
# ---------------------------------------------------------------------------

print("=" * 50)
print("REPRODUCIBILITY AUDIT")
print("=" * 50)
repro = audit_reproducibility(run_dir)
print(f"Status: {repro['status'].upper()}")
print(f"Fatal: {repro['fatal_count']}, Warnings: {repro['warning_count']}")
for c in repro["checks"]:
    icon = "PASS" if c["status"] == "pass" else "FAIL"
    print(f"  [{icon}] {c['id']}: {c['message'][:80]}")

# ---------------------------------------------------------------------------
# 2. Research Bias Audit
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print("RESEARCH BIAS AUDIT")
print("=" * 50)
bias = audit_research(run_dir)
print(f"Status: {bias['status'].upper()}")
print(f"Fatal: {bias['fatal_count']}, Warnings: {bias['warning_count']}")
for c in bias["checks"]:
    icon = "PASS" if c["status"] == "pass" else "FAIL"
    print(f"  [{c['severity']:7s}] {icon} {c['id']}: {c['message'][:80]}")

# ---------------------------------------------------------------------------
# 3. Robustness Tests
# ---------------------------------------------------------------------------

print(f"\n{'='*50}")
print("ROBUSTNESS TESTS")
print("=" * 50)
robust = run_robustness(run_dir)
print(f"Status: {robust['status'].upper()}")
print(f"Baseline Sharpe: {_format_metric(robust.get('baseline_sharpe'))}")
for t in robust["tests"]:
    icon = "PASS" if t["status"] == "pass" else ("FAIL" if t["status"] == "fail" else "WARN")
    print(f"  [{icon}] {t['name']}: {t.get('message', '')[:80]}")
    for key in ("baseline_sharpe", "oos_sharpe", "perturbed_sharpe"):
        if key in t:
            print(f"       {key}: {_format_metric(t[key])}")

# ---------------------------------------------------------------------------
# Save audit results alongside run artifacts
# ---------------------------------------------------------------------------

(run_dir / "reproducibility_audit.json").write_text(json.dumps(repro, indent=2))
(run_dir / "research_bias_audit.json").write_text(json.dumps(bias, indent=2))
(run_dir / "robustness.json").write_text(json.dumps(robust, indent=2, default=str))

print(f"\nAudit results written to {run_dir}/")

# ---------------------------------------------------------------------------
# CLI equivalents
# ---------------------------------------------------------------------------
print(f"""
{'='*60}
CLI equivalents:
  oxq audit reproducibility {run_dir}/
  oxq audit research {run_dir}/
  oxq robustness run {run_dir}/
{'='*60}
""")
