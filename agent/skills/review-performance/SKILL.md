---
name: review-performance
description: >-
  Review open-xquant backtest performance, audit findings, reports, and
  experiment comparisons; use when users ask whether a strategy worked or how
  to interpret results.
---

# Performance Reviewer

You explain results after artifacts and audits exist.

## Read Artifacts First

If `runs/<run_id>/research_report.md` exists, read it. If it does not exist,
read the run artifacts directly and use `write-research-report` before
presenting a final decision.
Do not replace the missing report with an ad hoc final decision; route through
`write-research-report` first.

Read metrics:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

run_dir = Path("runs/<run_id>")
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
cat experiments.jsonl
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
