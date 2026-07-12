from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import pytest

from oxq.cli import agent as agent_module
from oxq.cli import doctor as doctor_module


@pytest.mark.parametrize(
    "reader",
    [
        pytest.param(agent_module._status_payload, id="status"),
        pytest.param(doctor_module._check_agent, id="doctor"),
    ],
)
def test_agent_state_readers_wait_for_lifecycle_manifest_replacement(
    monkeypatch,
    tmp_path,
    reader: Callable[[], dict],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    manifest_path = agent_module.manifest_path()
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "agent_profile": "old",
                "targets": {"codex": {"installed": True, "skills": [], "agent_roles": []}},
            }
        ),
        encoding="utf-8",
    )
    original_lock = agent_module.agent_lifecycle_lock
    attempted = threading.Event()
    completed = threading.Event()
    results: list[dict] = []
    failures: list[BaseException] = []

    @contextmanager
    def observed_lock() -> Iterator[None]:
        attempted.set()
        with original_lock():
            yield

    def read_state() -> None:
        try:
            results.append(reader())
        except BaseException as exc:
            failures.append(exc)
        finally:
            completed.set()

    with original_lock():
        backup_path = manifest_path.with_suffix(".backup")
        manifest_path.replace(backup_path)
        monkeypatch.setattr(agent_module, "agent_lifecycle_lock", observed_lock)
        monkeypatch.setattr(doctor_module, "agent_lifecycle_lock", observed_lock)
        thread = threading.Thread(target=read_state)
        thread.start()
        assert attempted.wait(timeout=5)
        assert not completed.is_set()
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "agent_profile": "new",
                    "targets": {"codex": {"installed": True, "skills": [], "agent_roles": []}},
                }
            ),
            encoding="utf-8",
        )

    assert completed.wait(timeout=5)
    thread.join(timeout=5)
    assert failures == []
    assert len(results) == 1
    if reader is agent_module._status_payload:
        assert results[0]["agent_profile"] == "new"
    else:
        assert results[0]["status"] == "ok"
