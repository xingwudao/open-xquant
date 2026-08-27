# AGENTS.md

## Naming policy

Use only the current canonical names in code, docs, tests, contracts, examples,
and generated artifacts.

Canonical repository and product names:

- `open-xquant`
- `equant-py`
- `ebacktestcraft-py`

Canonical equant-py distribution names:

- `equant-core`
- `equant-ttr`
- `equant-classic`
- `equant-alpha101`
- `equant-gtalpha`
- `equant-candlesticks`
- `equant-factorcraft`
- `equant-datatools`

Canonical import package names:

- `equant`
- `ettr`
- `eclassic`
- `ealpha101`
- `egtalpha`
- `ecandlesticks`
- `efactorcraft`
- `edatatools`

Do not introduce legacy mixed-case names anywhere except in this policy file:

- `OpenXQuant`
- `Open XQuant`
- `Open-XQuant`
- `eQuant`
- `eTTR`
- `eClassic`
- `eAlpha101`
- `eGTAlpha`
- `eCandleSticks`
- `eFactorCraft`
- `eDataTools`
- `eBacktestCraft`
- `eBacktestCraft-Py`
- `eFinCharts`

Do not add tests, fixtures, contract examples, or docs that preserve those
legacy spellings as examples. If a naming rule needs to be documented, update
this file instead.
