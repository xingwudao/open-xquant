# /quant-backtest

Run a backtest from a validated strategy spec.

## Usage

```
/quant-backtest strategy_spec.yaml
```

## Steps

1. Verify spec passes `oxq spec validate strategy_spec.yaml`
2. `oxq strategy compile strategy_spec.yaml`
3. `oxq backtest run strategy_spec.yaml --out runs/auto`
4. Note the run directory path for audit and report steps

## Next

Use `/quant-audit runs/<run_id>/` to audit the results.
