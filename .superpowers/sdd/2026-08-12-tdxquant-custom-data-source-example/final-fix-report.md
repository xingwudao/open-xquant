# TdxQuant final fix-wave report

Date: 2026-08-12

## Scope

Changed only the approved final-review production/test/document files, plus
this required evidence report:

- `examples/custom_data_sources/tdxquant_downloader.py`
- `tests/examples/test_tdxquant_downloader.py`
- `examples/custom_data_sources/README.md`
- `docs/superpowers/specs/2026-08-12-tdxquant-custom-data-source-example-design.md`

The ignored implementation plan was not changed. No online smoke test ran.

## Root cause and TDD RED

Root cause review found that OHLC fields were converted by pandas directly,
which accepts Python booleans as `1.0` or `0.0`. Timeout validation used only
`timeout <= 0`, so `NaN` and positive infinity passed, while `URLError` with a
timeout reason took the generic connection-error branch. `strptime` accepted a
non-canonical single-digit month/day string.

Test additions were made before production changes. Command:

```text
uv run pytest -q tests/examples/test_tdxquant_downloader.py
```

RED output summary (before production fixes):

```text
collected 62 items
11 failed, 51 passed in 1.63s
```

The expected failures were:

- five `test_rejects_non_finite_or_non_positive_timeout` cases: old message
  lacked `finite`, and `NaN`/`+inf` were silently accepted;
- `test_rejects_invalid_date_range[2024-1-2-2024-01-03]`: old parser issued a
  request instead of rejecting the non-canonical date;
- direct timeout transport test: old message omitted endpoint;
- two wrapped timeout tests: old `URLError` branch reported connection failure;
- in-window and out-of-window boolean OHLC tests: old float conversion accepted
  the booleans.

The strengthened non-2xx response test sets `response.read` to raise
`AssertionError`; it remained green because status is checked before body read.

## Minimal implementation and GREEN

Implemented:

- `_ohlc_values`, which rejects `bool` before `float` conversion and wraps
  `TypeError`, `ValueError`, and `OverflowError` as `DownloadError`;
- canonical date validation by round-tripping parsed values with `isoformat()`;
- `math.isfinite(timeout)` plus positive timeout validation;
- endpoint/duration timeout errors for direct and `URLError.reason` timeouts;
- README production/atomicity warning;
- design synchronization for port `17709` and complete-response validation
  before range filtering.

GREEN command:

```text
uv run pytest -q tests/examples/test_tdxquant_downloader.py
```

GREEN output:

```text
collected 62 items
62 passed in 0.36s
```

## Full verification

```text
uv run ruff check examples/custom_data_sources tests/examples
All checks passed!

uv run mypy examples/custom_data_sources/tdxquant_downloader.py
Success: no issues found in 1 source file

uv run pytest -q tests/examples/test_tdxquant_downloader.py
62 passed in 0.36s

uv run pytest -qq
3738 passed, 4 skipped, 2 deselected, 271 warnings in 228.01s (0:03:48)

git diff --check
exit 0
```

## Self-review

- Boolean OHLC values are rejected before conversion; the regression covers an
  in-window `True` and an out-of-window `False`, proving full-response
  validation occurs before filtering.
- Parser failures are wrapped as `DownloadError`; finite numeric validation
  remains after frame creation for `NaN`/infinite numeric values.
- Timeout error messages include both the configured endpoint and duration for
  direct and wrapped timeout paths; non-timeout `URLError` behavior is kept.
- Canonical inputs remain canonical in the existing request/manifest contract.
- Non-2xx response handling remains before `read()`; the regression causes a
  body-read assertion if the ordering changes.
- README accurately discloses non-production overwrite/non-atomic behavior and
  names all four production requirements: temporary files, atomic replacement,
  incremental updates, and failure recovery.
- Committed design, not the ignored plan, now records explicit port `17709` and
  validation before date filtering.

## Concerns

None. The full suite emits existing deprecation warnings (271); no online smoke
test was run, by task requirement.

## Commit

One final-fix-wave commit with subject:

```text
fix: harden TdxQuant final review findings
```
