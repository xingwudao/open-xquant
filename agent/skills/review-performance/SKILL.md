---
name: review-performance
description: >-
  Review open-xquant backtest performance, audit findings, reports, and
  experiment comparisons; use when users ask whether a strategy worked or how
  to interpret results.
---

## Phase Path Containment Preflight

Before any phase artifact read, write, directory creation, command, or handoff,
run this preflight completely and block on any failure:

1. Read `.open-xquant/workspace.yaml`. Resolve `version_root` from
   `paths.versions_dir`, using `versions` only when that key is absent.
   Require `version_root` to be a safe workspace-relative path: reject an
   absolute path or any `..` path segment, resolve it canonically with
   symlinks, and require the result to stay inside the workspace.
2. Read `current.json`. For normal phase work, set `expected_version_id` to
   `current.json.active_version`; it must exist. Only a contract that
   explicitly owns cross-version inspection may instead set
   `expected_version_id` from the referenced version id for that historical
   read. This exception never permits active-version work to consume another
   version.
3. Set the intended version directory to
   `<version_root>/<expected_version_id>/` and resolve it canonically. Require
   the intended version directory to remain inside the canonical version root
   and workspace; otherwise treat it as a symlink escape. Read
   `version_manifest.json` only from that exact directory. The manifest
   `version_id` must equal `expected_version_id`; for normal phase work it
   therefore must also equal `current.json.active_version`.
4. Before using each required `phase_paths` value, require a non-empty
   workspace-relative string. Reject an absolute path and any `..` path
   segment. Resolve `<workspace>/<phase_path>` canonically, including existing
   symlink ancestors even when the leaf will be created, and require the target
   to be the intended version directory or a descendant of it. A symlink escape
   outside that directory is invalid.
5. On any identity or path failure, stop before phase artifact reads, writes,
   directory creation, commands, or handoffs. Do not normalize an unsafe path
   into acceptance and do not fall back to a default phase path.

Block examples when `expected_version_id` is `v001` include
`strategy_store/v001/../v002/04_spec_build`,
`strategy_store/v002/04_spec_build`, `/tmp/04_spec_build`, and
`strategy_store/v001/escape/04_spec_build` when `escape` is a symlink
whose target is outside the intended version directory. An allowed custom
nested phase path is
`strategy_store/v001/custom/phases/04_spec_build` when its canonical
target remains under the intended version directory.

For a new-version bootstrap, only `manage-strategy-version` may proceed before
the new manifest exists or the new id becomes active. It must apply the same
workspace-relative, traversal, canonical-containment, and symlink checks to
every constructed phase path before directory creation, then write a matching
manifest before publishing `current.json` last.

## Version Path Resolution

Before using any `<phase_paths.*>` or `<version_root>` placeholder, read
`.open-xquant/workspace.yaml`. Resolve `version_root` from
`paths.versions_dir`; use `versions` only when that key is absent. Require a
safe relative path whose resolved target stays inside the workspace. Then read
`<version_root>/<version_id>/version_manifest.json` and use its exact
`phase_paths` entry for each phase. For example, a configured root of
`research_versions` must resolve the spec-build phase to
`research_versions/v003/04_spec_build`; never redirect it to a default-root phase path.

Resolve `experiment_registry` independently from
`paths.experiment_registry`; use `experiments.jsonl` only when that key is
absent. Require a safe relative path whose resolved target stays inside the
workspace. Reject absolute paths, traversal outside the workspace, and symlink
escapes whose resolved target leaves the workspace. Use the resolved
`<experiment_registry>` for every experiment-registry read.

# Performance Reviewer

You explain results after artifacts and audits exist.

## Read Artifacts First

Resolve the run under the active version first:
`<phase_paths.09_backtests>/<run_id>/`. If
`<phase_paths.10_reports>/<run_id>/research_report.md` exists, read it.
If it does not exist, read the run artifacts directly and use
`write-research-report` before presenting a final decision.
Do not replace the missing report with an ad hoc final decision; route through
`write-research-report` first.

Read metrics:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

run_dir = Path("<phase_paths.09_backtests>/<run_id>")
metrics = json.loads((run_dir / "metrics.json").read_text())
for key in ["total_return", "sharpe_ratio", "max_drawdown", "trade_count"]:
    print(key, metrics.get(key))
PY
```

Read trades and equity only after confirming `metrics.json` exists.

## Review Order

1. Reproducibility audit status.
2. Research audit fatal findings.
3. Research audit warnings.
4. Metrics profile and metric assumptions.
5. Execution assumptions, calendar, lot size, and cash return.
6. Robustness status, including IS/OOS diff, perturbation, and regimes.
7. OOS trade count and test-period coverage.
8. Return, Sharpe, drawdown, and benchmark comparison.
9. Whether the report decision is justified.

## Decision Language

Use conservative labels:

- `REJECT`: fatal audit, invalid spec, missing data, or clearly poor OOS
- `WATCHLIST`: no fatal audit, but warnings or weak robustness remain
- `PAPER TRADING CANDIDATE`: audit clean enough, OOS plausible, and user
  accepts remaining risks

Do not use `PAPER TRADING CANDIDATE` when research audit has fatal findings or
when data provenance is unclear.

## Compare Runs

```bash
cat <experiment_registry>
```

If comparing multiple runs, normalize:

- same data source
- same time period
- same benchmark
- same fee and slippage assumptions
- same metrics profile and metric assumptions
- same execution price, calendar, lot size, and cash return assumptions
- same universe

## Red Lines

- Do not explain away negative Sharpe or severe drawdown.
- Do not rank strategies that used different data or costs without stating it.
- Do not make investment advice; report research evidence and limitations.
