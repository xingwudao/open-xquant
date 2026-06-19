# /quant-audit

Audit a backtest run for bias and reproducibility.

## Usage

```
/quant-audit runs/<run_id>/
```

## Steps

1. `oxq audit reproducibility runs/<run_id>/` — verify hash consistency
2. `oxq audit research runs/<run_id>/` — check for biases
3. `oxq robustness run runs/<run_id>/` — cost, IS/OOS, perturbation, and regime checks

## Output

- reproducibility_audit.json
- research_bias_audit.json
- robustness.json

## Review

- Confirm `execution_assumptions.json` is present when generated.
- Confirm `metrics.json` records metrics profile assumptions.
- Preserve fatal, warning, fragile, and error statuses.

## Next

Use `/quant-report runs/<run_id>/` to generate the research report.
