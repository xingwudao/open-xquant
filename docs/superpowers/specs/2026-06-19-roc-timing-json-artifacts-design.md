# ROC Timing JSON Artifacts Design

## Goal

Close the remaining acceptance gaps for xingwudao/open-xquant#6 and xingwudao/open-xquant#7 before opening the PR from `codex-xquant-roc-timing-artifacts`.

## Scope

- Treat deterministic in-repository CSI300-like fixtures as the strict-tolerance baseline for #6.
- Keep the existing `ROCTiming` and `SignalPosition` runtime model, and add missing coverage for rolling quantile timing and end-to-end generated specs.
- Add machine-readable `oxq backtest run --json` output for #7.
- Make `target_weights.csv` discoverable from JSON output and verified by reproducibility audit.

## Design

`ROCTiming` remains a tri-state signal: `1.0` means BUY, `0.0` means SELL, and `-1.0` means HOLD. `SignalPosition` continues to preserve the current position across HOLD bars and move to cash on SELL. End-to-end tests will build fixed-threshold and rolling-quantile specs, run them on deterministic market data, and compare `target_weights.csv` against expected baseline rows at strict tolerance.

`oxq backtest run --json` will suppress human progress text and emit one JSON object. The payload includes `run_id`, `run_dir`, artifact paths, summary metrics, validation warnings, and validation errors. Human output remains unchanged when `--json` is absent.

`artifact_hashes.json` already includes `target_weights.csv`; reproducibility audit will now require and verify it for schema version 2 runs.

## Testing

- Unit test rolling quantile `ROCTiming` output.
- End-to-end compile tests for fixed and rolling ROC timing specs against deterministic baselines.
- CLI test for `backtest run --json`.
- Reproducibility audit test that tampering with `target_weights.csv` fails.
