---
name: backtest-runner
description: >-
  Compatibility skill for old prompts that ask to run a backtest; redirect to
  strategy-builder for spec creation and audited backtest workflow.
---

# Backtest Runner

This skill is kept for backward compatibility.

Do not implement a separate backtest flow here. Load
`agent/skills/strategy-builder.md` and follow its validated
`strategy_spec.yaml` workflow:

```text
idea -> spec init -> spec validate -> strategy compile -> backtest
     -> reproducibility audit -> research audit -> robustness -> report
```

The redirected workflow inherits current runtime assumptions from
`strategy-builder.md`, including explicit execution semantics, metrics
profiles, supported calendars, and robustness artifacts.

If the user already has a validated spec, skip the idea-building phases in
`strategy-builder.md` and run only the compile/backtest/audit/report steps.
