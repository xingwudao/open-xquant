from __future__ import annotations

from pathlib import Path

MONITOR_SKILL = Path("agent/skills/monitor-strategy-run/SKILL.md")
MONITOR_ROLE = Path("agent/roles/oxq-monitor-worker.md")


def test_monitor_skill_defines_final_integrity_refresh_ordering() -> None:
    text = MONITOR_SKILL.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "canonical post-monitor integrity refresh" in normalized
    assert "1. Publish `reproducibility_audit.json`" in text
    assert "2. Publish `research_bias_audit.json`" in text
    assert "3. Run robustness, which self-publishes `robustness.json`." in text
    assert "4. Run `oxq experiment add` as the final run-package mutation." in text
    assert "`reproducibility_audit.json`, `research_bias_audit.json`, and `robustness.json`" in normalized
    assert "exactly one current entry in `<phase_paths.09_backtests>/run_digests.jsonl`" in normalized
    assert "Do not write another run artifact after this refresh" in text


def test_monitor_role_requires_final_integrity_refresh_before_report_handoff() -> None:
    text = MONITOR_ROLE.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "canonical post-monitor integrity refresh" in normalized
    assert "Run experiment registration last among monitor mutations" in text
    assert "Do not mutate the run package after the refresh" in text
