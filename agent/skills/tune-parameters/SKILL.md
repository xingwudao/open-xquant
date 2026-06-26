---
name: tune-parameters
description: >-
  Tune open-xquant strategy parameters with grid search, walk-forward
  validation, time-series CV, and overfitting checks; use when users ask to
  optimize parameters.
---

# Parameter Tuner

You search parameters without turning in-sample luck into a claim.

## Preconditions

- The base strategy logic must already be clear.
- The base spec or SDK strategy must pass validation.
- Data must cover both training and OOS periods.
- The user must approve the metric and search ranges.

Warn when the total grid exceeds 100 combinations.

## SDK Pattern

The optimizer API works on `Strategy` objects, not directly on a YAML spec.
If starting from a spec, compile it first or follow `examples/modules/07_optimize.py`.

```python
from oxq.optimize.paramset import ParameterSet
from oxq.optimize.search import GridSearch

paramset = ParameterSet(name="sma_tuning")
paramset.add("sma_10", "period", list(range(5, 30, 5)))
paramset.add("sma_50", "period", list(range(30, 100, 20)))
paramset.add_constraint("sma_10.period < sma_50.period")

search = GridSearch(paramset).run(
    strategy=strategy,
    market=market,
    broker_factory=broker_factory,
    start="2018-01-01",
    end="2021-12-31",
    metric="sharpe_ratio",
)
```

Parameter component names must match strategy component names such as
indicator aliases in `required_indicators`. If a name does not match, the
parameter has no effect.

## Walk-Forward Required

```python
from oxq.optimize.walk_forward import WalkForward

wf = WalkForward(
    paramset=paramset,
    train_period="2Y",
    test_period="1Y",
    step="1Y",
)
wf_result = wf.run(
    strategy=strategy,
    market=market,
    broker_factory=broker_factory,
    start="2018-01-01",
    end="2024-12-31",
    metric="sharpe_ratio",
)
print(wf_result.deterioration())
```

## Time-Series CV

```python
from oxq.optimize.validation import TimeSeriesCV

cv = TimeSeriesCV(n_splits=4, expanding=True)
cv_result = cv.cross_validate(
    strategy=strategy,
    market=market,
    broker_factory=broker_factory,
    start="2018-01-01",
    end="2024-12-31",
    paramset=paramset,
    metric="sharpe_ratio",
)
```

## Overfit Signals

- IS Sharpe far above OOS Sharpe
- best parameters lie on search boundary
- OOS return or Sharpe turns negative
- tiny parameter changes destroy performance
- selected configuration has very few trades

Before comparing parameter sets, normalize metrics profile, annualization,
risk-free rate, execution price, market calendar, lot size, costs, and cash
return assumptions. Use the robustness runner's parameter perturbation and
IS/OOS metric diff outputs when they are available.

## Red Lines

- Do not report "best parameters" without OOS validation.
- Do not expand the search range after seeing OOS just to rescue a result.
- Do not tune on the final test period.
