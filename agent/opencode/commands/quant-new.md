# /quant-new

Start a new quantitative research project.

## Usage

```
/quant-new "SMA crossover strategy with weekly rebalance"
```

## Steps

1. `oxq spec init "<idea>" --out strategy_spec.yaml`
2. Edit strategy_spec.yaml with proper fields
3. `oxq spec validate strategy_spec.yaml`
4. Fix any errors and re-validate

## Next

After validation passes, use `/quant-backtest` to run the backtest.
