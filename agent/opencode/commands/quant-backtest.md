# /quant-backtest

Run a backtest from a validated strategy spec.

## Usage

```
/quant-backtest strategy_spec.yaml
```

## Steps

1. Verify spec passes `oxq spec validate strategy_spec.yaml`
2. `oxq strategy compile strategy_spec.yaml`
3. `oxq backtest run strategy_spec.yaml --out runs/auto --json > backtest.json`
4. Read `run_dir` and artifact paths from `backtest.json`

## Next

Use `/quant-audit runs/<run_id>/` to audit the results.
