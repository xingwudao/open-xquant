from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from oxq.run_digests import require_current_run_digest
from oxq.spec.compiler import compile_run
from oxq.spec.schema import StrategySpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MONITOR_SKILL = Path("agent/skills/monitor-strategy-run/SKILL.md")
MONITOR_ROLE = Path("agent/roles/oxq-monitor-worker.md")
COORDINATOR_ROLE = Path("agent/roles/oxq-coordinator.md")
ARCHITECTURE_DOC = Path("docs/architecture.md")
GOVERNANCE_DOC = Path("docs/strategy-workflow-artifact-governance.md")
MONITOR_COMMAND_CONTRACTS = (
    MONITOR_SKILL,
    MONITOR_ROLE,
    ARCHITECTURE_DOC,
    GOVERNANCE_DOC,
)
MONITOR_GUIDANCE = (
    *tuple(sorted(Path("agent").rglob("*.md"))),
    ARCHITECTURE_DOC,
    GOVERNANCE_DOC,
)
CANONICAL_COMMANDS = (
    'uv run oxq audit reproducibility "$RUN_DIR" --json --publish',
    'uv run oxq audit research "$RUN_DIR" --json --publish',
    'uv run oxq robustness run "$RUN_DIR" --json',
)
GOVERNED_MONITOR_PREFIXES = (
    "oxq audit reproducibility",
    "oxq audit research",
    "oxq robustness run",
)
SHELL_REDIRECTION = re.compile(r"(?:^|\s)>{1,2}\s*\S")


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return " ".join(_text(path).split())


def _governed_run(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    backtest_dir = workspace / "versions/v001/09_backtests"
    data_dir = workspace / "data"
    config_dir = workspace / ".open-xquant"
    for path in (backtest_dir, data_dir, config_dir):
        path.mkdir(parents=True)

    (config_dir / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {"versions_dir": "versions"},
                "workflow": {"layout": "version_governed"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (workspace / "current.json").write_text(
        json.dumps({"active_version": "v001"}),
        encoding="utf-8",
    )
    (workspace / "versions/v001/version_manifest.json").write_text(
        json.dumps(
            {
                "version_id": "v001",
                "phase_paths": {
                    "09_backtests": "versions/v001/09_backtests",
                },
            }
        ),
        encoding="utf-8",
    )

    dates = pd.bdate_range("2024-01-02", "2024-01-12", tz="UTC")
    pd.DataFrame(
        {
            "open": range(100, 100 + len(dates)),
            "high": range(101, 101 + len(dates)),
            "low": range(99, 99 + len(dates)),
            "close": range(100, 100 + len(dates)),
            "volume": [1000] * len(dates),
        },
        index=dates,
    ).to_parquet(data_dir / "SPY.parquet")
    spec = StrategySpec.template(
        strategy_id="round_21_monitor_contract",
        hypothesis="canonical monitor publishers keep the run digest current",
    )
    spec.validation.train_period = ["2024-01-02", "2024-01-05"]
    spec.validation.test_period = ["2024-01-08", "2024-01-12"]
    _, run_dir = compile_run(spec, data_dir=str(data_dir), out_dir=backtest_dir)
    return run_dir


def test_monitor_guidance_requires_canonical_publishers_without_redirection() -> None:
    for path in MONITOR_COMMAND_CONTRACTS:
        text = _text(path)
        for command in CANONICAL_COMMANDS:
            assert command in text, (path, command)

    for path in MONITOR_GUIDANCE:
        logical_lines = _text(path).replace("\\\n", " ").splitlines()
        for line in logical_lines:
            if any(prefix in line for prefix in GOVERNED_MONITOR_PREFIXES):
                assert not SHELL_REDIRECTION.search(line), (path, line)

    for path in (MONITOR_SKILL, MONITOR_ROLE, GOVERNANCE_DOC):
        normalized = _normalized(path)
        assert "`--json` is response formatting" in normalized, path
        assert "`--publish` is the audit publication contract" in normalized, path
        assert "robustness needs no redirection or extra publish flag" in normalized, path

    coordinator = _normalized(COORDINATOR_ROLE)
    assert "canonical monitor publishers" in coordinator
    assert "current run digest" in coordinator
    assert "read-only report handoff" in coordinator


def test_monitor_validates_manifest_owned_spec_audit_and_binds_run_copy() -> None:
    text = _normalized(MONITOR_SKILL)

    assert "`<phase_paths.06_spec_audit>/spec_audit.json` is the canonical SPEC audit" in text
    assert "uv run oxq spec-audit validate <phase_paths.06_spec_audit>/spec_audit.json" in text
    assert "run directory is an attached provenance copy" in text
    assert "compare its bytes" in text
    assert "hash" in text
    assert "uv run oxq spec-audit validate <phase_paths.09_backtests>/<run_id>/spec_audit.json" not in text


def test_exact_monitor_commands_keep_governed_run_digest_current(tmp_path: Path) -> None:
    run_dir = _governed_run(tmp_path)
    env = {**os.environ, "RUN_DIR": str(run_dir)}
    expected_artifacts = (
        "reproducibility_audit.json",
        "research_bias_audit.json",
        "robustness.json",
    )

    for command, artifact_name in zip(CANONICAL_COMMANDS, expected_artifacts, strict=True):
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert isinstance(json.loads(result.stdout), dict)
        assert isinstance(json.loads((run_dir / artifact_name).read_text(encoding="utf-8")), dict)
        require_current_run_digest(run_dir)
