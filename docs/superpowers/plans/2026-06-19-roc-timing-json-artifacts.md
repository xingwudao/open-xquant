# ROC Timing JSON Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the issue #6 and #7 acceptance gaps and open a PR that can close both issues.

**Architecture:** Keep the existing runtime components and add behavior at their boundaries: tests define `ROCTiming`/`SignalPosition` target-weight alignment, CLI JSON wraps `compile_run`, and reproducibility audit validates the emitted target-weight artifact. The PR body can use cross-repo closing keywords after these checks pass.

**Tech Stack:** Python, Click, pandas, pytest, existing open-xquant spec compiler and audit modules.

---

### Task 1: Test Missing Acceptance

**Files:**
- Modify: `tests/signals/test_roc_timing.py`
- Modify: `tests/spec/test_target_weight_artifact.py`
- Modify: `tests/cli/test_main.py`

- [ ] Add a rolling quantile ROC timing unit test.
- [ ] Add fixed-threshold and rolling-quantile compile-run tests that compare `target_weights.csv` against deterministic baseline rows.
- [ ] Add a CLI test that `oxq backtest run --json` returns JSON with `run_id`, `run_dir`, artifact paths, summary metrics, warnings, and errors.
- [ ] Add an audit test that tampering with `target_weights.csv` fails.
- [ ] Run the new tests and verify they fail for missing behavior.

### Task 2: Implement CLI JSON and Audit Support

**Files:**
- Modify: `src/oxq/cli/main.py`
- Modify: `src/oxq/spec/compiler.py`
- Modify: `src/oxq/audit/reproducibility.py`

- [ ] Add a helper that builds backtest JSON output from `RunResult`, `run_dir`, artifacts, and validation.
- [ ] Add `--json` to `backtest run` and emit only JSON in JSON mode.
- [ ] Include `target_weights.csv` in the run artifact list.
- [ ] Require `target_weights.csv` in schema version 2 reproducibility audit.
- [ ] Run targeted tests and verify they pass.

### Task 3: Verify, Push, and Open PR

**Files:**
- Local git branch and GitHub PR metadata.

- [ ] Run targeted pytest for changed behavior.
- [ ] Run relevant broader tests if targeted tests pass.
- [ ] Inspect `git diff` and `git status`.
- [ ] Commit the completed changes.
- [ ] Push the current branch.
- [ ] Create a PR with `Closes xingwudao/open-xquant#6` and `Closes xingwudao/open-xquant#7`.
