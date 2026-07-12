from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from oxq import process_lock, run_digests
from oxq.core.engine import Engine
from oxq.core.types import Portfolio
from oxq.observe import experiment_registry
from oxq.observe.experiment_registry import add_experiment
from oxq.portfolio.analytics import RunResult
from oxq.process_lock import ProcessFileLock
from oxq.run_digests import (
    multi_run_digest_read_transaction,
    publish_run_artifacts,
    require_current_run_digest,
    update_artifact_hashes_and_run_digest,
)
from oxq.spec.compiler import _write_artifacts
from oxq.spec.schema import StrategySpec


def _case_insensitive_alias(path: Path) -> Path:
    alias = path.with_name(path.name.swapcase())
    try:
        aliases_same_entry = alias.exists() and os.path.samefile(path, alias)
    except OSError:
        aliases_same_entry = False
    if not aliases_same_entry:
        pytest.skip("filesystem does not expose case-insensitive aliases")
    return alias


def test_add_experiment_runs_and_persists_research_audit(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"

    spec = StrategySpec.template(strategy_id="audit_registry", hypothesis="audit status is preserved")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "audit_registry", "run_id": "run_1", "trade_count": 12, "max_drawdown": -0.10}),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash(), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    audit_path = run_dir / "research_bias_audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert entry["audit_status"] == audit["status"]
    assert entry["audit_status"] != "unknown"
    assert not (run_dir / "artifact_hashes.json").exists()
    assert not (run_dir.parent / "run_digests.jsonl").exists()


def test_add_experiment_rejects_metrics_strategy_id_mismatch_before_mutation(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    spec = StrategySpec.template(strategy_id="spec_identity", hypothesis="identity is immutable")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "metrics_identity", "run_id": "run_1"}),
        encoding="utf-8",
    )
    original_registry = b'{"experiment_id":"existing"}\n'
    original_bias = b'{"status":"existing"}\n'
    registry_path.write_bytes(original_registry)
    (run_dir / "research_bias_audit.json").write_bytes(original_bias)

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert entry == {
        "error": "metrics.json strategy_id must match strategy_spec.yaml strategy_id: spec_identity"
    }
    assert registry_path.read_bytes() == original_registry
    assert (run_dir / "research_bias_audit.json").read_bytes() == original_bias


def test_add_experiment_rejects_noncanonical_spec_hash_before_mutation(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    spec = StrategySpec.template(strategy_id="spec_identity", hypothesis="identity is immutable")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text("sha256:" + "0" * 16 + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "spec_identity", "run_id": "run_1"}),
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert entry == {"error": "spec_hash.txt must match the canonical strategy_spec.yaml hash"}
    assert not registry_path.exists()
    assert not (run_dir / "research_bias_audit.json").exists()


def test_add_experiment_rejects_unparseable_strategy_spec_before_mutation(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    (run_dir / "strategy_spec.yaml").write_text("research: [\n", encoding="utf-8")
    (run_dir / "spec_hash.txt").write_text("sha256:" + "0" * 16 + "\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "spec_identity", "run_id": "run_1"}),
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert "strategy_spec.yaml could not be parsed" in entry["error"]
    assert not registry_path.exists()
    assert not (run_dir / "research_bias_audit.json").exists()


def test_add_experiment_refreshes_monitor_artifact_hashes_and_run_digest(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    spec = StrategySpec.template(strategy_id="monitor_refresh", hypothesis="monitor outputs stay governed")
    _write_artifacts(
        spec,
        RunResult(
            portfolio=Portfolio(cash=Decimal("100000")),
            trades=[],
            equity_curve=[
                ("2024-01-02", 100_000.0),
                ("2024-01-03", 100_000.0),
                ("2024-01-04", 90_000.0),
            ],
            mktdata={},
        ),
        run_dir,
        Engine(),
    )
    (run_dir / "reproducibility_audit.json").write_text(
        json.dumps({"status": "pass", "fatal_count": 0, "warning_count": 0}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "robustness.json").write_text(
        json.dumps({"status": "warn", "tests": []}) + "\n",
        encoding="utf-8",
    )
    (run_dir.parent / "run_digests.jsonl").write_text(
        json.dumps({"run_id": "run_1", "artifact_hashes": "sha256:" + "0" * 16}) + "\n",
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert "error" not in entry
    hashes = json.loads((run_dir / "artifact_hashes.json").read_text(encoding="utf-8"))
    for artifact_name in (
        "reproducibility_audit.json",
        "research_bias_audit.json",
        "robustness.json",
    ):
        assert hashes[artifact_name] == _file_hash(run_dir / artifact_name)
    from oxq.run_digests import require_current_run_digest

    require_current_run_digest(run_dir)


def test_add_experiment_uses_windows_process_lock_without_importing_fcntl(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeMsvcrt:
        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            self.operations: list[int] = []

        def locking(self, descriptor: int, operation: int, length: int) -> None:
            assert descriptor >= 0
            assert length == 1
            self.operations.append(operation)

    fake_msvcrt = FakeMsvcrt()
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    _write_identity_artifacts(run_dir, "windows_registry")
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "strategy_id": "windows_registry",
                "run_id": "run_1",
                "trade_count": 12,
                "max_drawdown": -0.1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )
    monkeypatch.setattr(process_lock, "_platform_name", lambda: "win32")
    monkeypatch.setattr(
        process_lock.importlib,
        "import_module",
        lambda name: fake_msvcrt if name == "msvcrt" else (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    def synthetic_location_identity(path) -> str:
        return str(Path(path).resolve(strict=False)).casefold()

    monkeypatch.setattr(experiment_registry, "stable_path_location_identity", synthetic_location_identity)
    monkeypatch.setattr(run_digests, "stable_path_location_identity", synthetic_location_identity)
    monkeypatch.setattr(experiment_registry, "stable_filesystem_identity", synthetic_location_identity)
    monkeypatch.setattr(run_digests, "stable_filesystem_identity", synthetic_location_identity)
    monkeypatch.setitem(sys.modules, "fcntl", None)

    entry = add_experiment(run_dir, registry_path=tmp_path / "experiments.jsonl")

    assert "error" not in entry
    assert fake_msvcrt.operations
    assert fake_msvcrt.operations.count(fake_msvcrt.LK_NBLCK) == fake_msvcrt.operations.count(
        fake_msvcrt.LK_UNLCK
    )


def test_concurrent_add_experiment_serializes_complete_monitor_sequence(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    spec = StrategySpec.template(
        strategy_id="concurrent_monitor",
        hypothesis="monitor publication stays serialized",
    )
    _write_artifacts(
        spec,
        RunResult(
            portfolio=Portfolio(cash=Decimal("100000")),
            trades=[],
            equity_curve=[("2024-01-02", 100_000.0), ("2024-01-03", 90_000.0)],
            mktdata={},
        ),
        run_dir,
        Engine(),
    )
    update_artifact_hashes_and_run_digest(run_dir, lambda hashes: None)
    first_audit_entered = threading.Event()
    allow_first_audit = threading.Event()
    second_lock_attempted = threading.Event()
    second_audit_entered = threading.Event()
    audit_guard = threading.Lock()
    audit_calls = 0
    active_audits = 0
    overlap_detected = False
    failures: list[BaseException] = []
    entries: list[dict] = []
    original_lock = ProcessFileLock

    @contextmanager
    def observed_registry_lock(path):
        if threading.current_thread().name == "monitor-second":
            second_lock_attempted.set()
        with original_lock(path):
            yield

    def controlled_audit(_run_path):
        nonlocal active_audits, audit_calls, overlap_detected
        with audit_guard:
            audit_calls += 1
            generation = audit_calls
            active_audits += 1
            overlap_detected = overlap_detected or active_audits > 1
        try:
            if generation == 1:
                first_audit_entered.set()
                assert allow_first_audit.wait(timeout=5)
            else:
                second_audit_entered.set()
            return {"status": "pass", "generation": generation}
        finally:
            with audit_guard:
                active_audits -= 1

    def add() -> None:
        try:
            entries.append(add_experiment(run_dir, registry_path=registry_path))
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(experiment_registry, "ProcessFileLock", observed_registry_lock, raising=False)
    monkeypatch.setattr("oxq.audit.research_bias.audit_research", controlled_audit)
    first = threading.Thread(target=add, name="monitor-first")
    second = threading.Thread(target=add, name="monitor-second")
    first.start()
    assert first_audit_entered.wait(timeout=5)
    second.start()
    try:
        assert second_lock_attempted.wait(timeout=5)
        assert not second_audit_entered.is_set()
    finally:
        allow_first_audit.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert overlap_detected is False
    assert len(entries) == 2
    assert len(registry_path.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads((run_dir / "research_bias_audit.json").read_text(encoding="utf-8"))[
        "generation"
    ] == 2
    require_current_run_digest(run_dir)


def test_add_experiment_entry_uses_artifacts_from_locked_run_generation(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    initial_spec = StrategySpec.template(
        strategy_id="initial_generation",
        hypothesis="registry initially sees this generation",
    )
    _write_artifacts(
        initial_spec,
        RunResult(
            portfolio=Portfolio(cash=Decimal("100000")),
            trades=[],
            equity_curve=[("2024-01-02", 100_000.0), ("2024-01-03", 90_000.0)],
            mktdata={},
        ),
        run_dir,
        Engine(),
    )

    transaction_entered = threading.Event()
    allow_registry_transaction = threading.Event()
    original_transaction = run_digests.run_digest_transaction
    paused = False

    @contextmanager
    def pause_before_registry_transaction(path):
        nonlocal paused
        if threading.current_thread().name == "registry-worker" and not paused:
            paused = True
            transaction_entered.set()
            assert allow_registry_transaction.wait(timeout=5)
        with original_transaction(path):
            yield

    monkeypatch.setattr(run_digests, "run_digest_transaction", pause_before_registry_transaction)
    entries: list[dict] = []
    failures: list[BaseException] = []

    def register() -> None:
        try:
            entries.append(add_experiment(run_dir, registry_path=registry_path))
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=register, name="registry-worker")
    worker.start()
    assert transaction_entered.wait(timeout=5)

    published_spec = StrategySpec.template(
        strategy_id="locked_generation",
        hypothesis="registry must use this locked generation",
    )
    published_metrics = {
        "strategy_id": published_spec.strategy_id,
        "run_id": run_dir.name,
        "trade_count": 7,
        "max_drawdown": -0.2,
    }
    published_hash = tmp_path / "published-spec-hash.txt"
    published_hash.write_text(published_spec.compute_hash() + "\n", encoding="utf-8")
    try:
        publish_run_artifacts(
            run_dir,
            {
                "strategy_spec.yaml": yaml.safe_dump(
                    published_spec.to_dict(),
                    sort_keys=False,
                ).encode(),
                "metrics.json": (json.dumps(published_metrics) + "\n").encode(),
            },
            replacement_paths={"spec_hash.txt": published_hash},
        )
    finally:
        allow_registry_transaction.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []
    assert len(entries) == 1
    registry_entry = json.loads(registry_path.read_text(encoding="utf-8"))
    locked_spec = StrategySpec.from_yaml(run_dir / "strategy_spec.yaml")
    locked_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    locked_spec_hash = (run_dir / "spec_hash.txt").read_text(encoding="utf-8").strip()
    assert registry_entry == entries[0]
    assert registry_entry["strategy_id"] == locked_spec.strategy_id == "locked_generation"
    assert registry_entry["metrics"] == locked_metrics == published_metrics
    assert registry_entry["spec_hash"] == locked_spec_hash == locked_spec.compute_hash()
    require_current_run_digest(run_dir)


def test_add_experiment_refresh_failure_restores_prior_generation(tmp_path, monkeypatch) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "refresh_rollback")
    snapshots = _transaction_snapshots(run_dir, registry_path)

    def fail_refresh(*_args, **_kwargs):
        raise OSError("injected refresh failure")

    monkeypatch.setattr(experiment_registry, "refresh_monitor_integrity", fail_refresh)

    result = add_experiment(run_dir, registry_path=registry_path)

    assert result == {"error": "monitor artifact integrity refresh failed: injected refresh failure"}
    assert _transaction_snapshots(run_dir, registry_path) == snapshots


def test_add_experiment_append_failure_restores_prior_generation(tmp_path, monkeypatch) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "append_rollback")
    snapshots = _transaction_snapshots(run_dir, registry_path)

    def fail_append(*_args, **_kwargs):
        raise OSError("injected registry append failure")

    monkeypatch.setattr(
        experiment_registry,
        "_append_registry_entry_locked",
        fail_append,
        raising=False,
    )

    result = add_experiment(run_dir, registry_path=registry_path)

    assert result == {"error": "experiment registry append failed: injected registry append failure"}
    assert _transaction_snapshots(run_dir, registry_path) == snapshots


def test_add_experiment_rejects_symlinked_registry_before_run_mutation(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "registry_symlink")
    snapshots = _transaction_snapshots(run_dir, registry_path)
    outside = tmp_path / "outside-experiments.jsonl"
    outside.write_bytes(registry_path.read_bytes())
    registry_path.unlink()
    registry_path.symlink_to(outside)

    result = add_experiment(run_dir, registry_path=registry_path)

    assert result == {"error": f"experiment registry must be a regular, non-symlink file: {registry_path}"}
    assert _transaction_snapshots(run_dir, registry_path, follow_registry=False)[:3] == snapshots[:3]
    assert outside.read_bytes() == snapshots[3]


@pytest.mark.parametrize(
    "managed_name",
    [
        "research_bias_audit.json",
        "metrics.json",
        "artifact_hashes.json",
        "run_digests.jsonl",
    ],
)
def test_add_experiment_rejects_registry_alias_to_managed_run_state_before_mutation(
    tmp_path,
    managed_name: str,
) -> None:
    strategy_id = f"registry_alias_{managed_name.replace('.', '_')}"
    run_dir, _registry_path = _write_transactional_run(tmp_path, strategy_id)
    managed_paths = {
        "research_bias_audit.json": run_dir / "research_bias_audit.json",
        "metrics.json": run_dir / "metrics.json",
        "artifact_hashes.json": run_dir / "artifact_hashes.json",
        "run_digests.jsonl": run_dir.parent / "run_digests.jsonl",
    }
    snapshots = {name: path.read_bytes() for name, path in managed_paths.items()}
    registry_path = managed_paths[managed_name]
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    journal_path = registry_path.with_suffix(registry_path.suffix + ".transaction.json")
    lock_existed = lock_path.exists()

    result = add_experiment(run_dir, registry_path=registry_path)

    assert "error" in result and "overlap" in result["error"]
    assert {name: path.read_bytes() for name, path in managed_paths.items()} == snapshots
    assert lock_path.exists() is lock_existed
    assert not journal_path.exists()


def test_add_experiment_rejects_hard_link_registry_alias_before_lock_creation(tmp_path) -> None:
    run_dir, _registry_path = _write_transactional_run(tmp_path, "hard_link_registry_alias")
    metrics_path = run_dir / "metrics.json"
    registry_path = tmp_path / "registry-hard-link.jsonl"
    os.link(metrics_path, registry_path)
    snapshots = _transaction_snapshots(run_dir, _registry_path)

    result = add_experiment(run_dir, registry_path=registry_path)

    assert "error" in result and "overlap" in result["error"]
    assert _transaction_snapshots(run_dir, _registry_path) == snapshots
    assert not registry_path.with_suffix(registry_path.suffix + ".lock").exists()
    assert not registry_path.with_suffix(registry_path.suffix + ".transaction.json").exists()


@pytest.mark.parametrize(
    ("internal_suffix", "managed_name"),
    [
        (".lock", "metrics.json"),
        (".transaction.json", "artifact_hashes.json"),
    ],
)
def test_add_experiment_rejects_registry_internal_alias_to_run_artifact(
    tmp_path,
    internal_suffix: str,
    managed_name: str,
) -> None:
    strategy_id = f"registry_internal_{managed_name.replace('.', '_')}"
    run_dir, registry_path = _write_transactional_run(tmp_path, strategy_id)
    internal_path = registry_path.with_suffix(registry_path.suffix + internal_suffix)
    os.link(run_dir / managed_name, internal_path)
    snapshots = _transaction_snapshots(run_dir, registry_path)

    result = add_experiment(run_dir, registry_path=registry_path)

    assert "error" in result and "overlap" in result["error"]
    assert _transaction_snapshots(run_dir, registry_path) == snapshots
    assert os.path.samefile(internal_path, run_dir / managed_name)


def test_add_experiment_rejects_missing_registry_location_inside_run(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "registry_inside_run")
    snapshots = _transaction_snapshots(run_dir, registry_path)
    inside_registry = run_dir / "experiments.jsonl"

    result = add_experiment(run_dir, registry_path=inside_registry)

    assert "error" in result and "overlap" in result["error"]
    assert _transaction_snapshots(run_dir, registry_path) == snapshots
    assert not inside_registry.exists()
    assert not inside_registry.with_suffix(".jsonl.lock").exists()
    assert not inside_registry.with_suffix(".jsonl.transaction.json").exists()


def test_add_experiment_rejects_nested_registry_before_creating_parent_inside_run(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "nested_registry_inside_run")
    snapshots = _transaction_snapshots(run_dir, registry_path)
    nested_parent = run_dir / "new-registry-parent"
    inside_registry = nested_parent / "experiments.jsonl"

    result = add_experiment(run_dir, registry_path=inside_registry)

    assert "error" in result and "overlap" in result["error"]
    assert _transaction_snapshots(run_dir, registry_path) == snapshots
    assert not nested_parent.exists()


def test_add_experiment_normalizes_overlap_before_walking_missing_dotdot_parent(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "dotdot_registry_inside_run")
    snapshots = _transaction_snapshots(run_dir, registry_path)
    missing_parent = tmp_path / "must-not-be-created"
    inside_registry = missing_parent / ".." / run_dir.name / "rejected-experiments.jsonl"

    result = add_experiment(run_dir, registry_path=inside_registry)

    assert "error" in result and "overlap" in result["error"]
    assert _transaction_snapshots(run_dir, registry_path) == snapshots
    assert not missing_parent.exists()
    assert not (run_dir / "rejected-experiments.jsonl").exists()


def test_add_experiment_rejects_registry_alias_to_digest_atomic_temp_namespace(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "registry_digest_temp_alias")
    snapshots = _transaction_snapshots(run_dir, registry_path)
    temp_alias = run_dir.parent / ".run_digests.jsonl.collision.tmp"

    result = add_experiment(run_dir, registry_path=temp_alias)

    assert "error" in result and "overlap" in result["error"]
    assert _transaction_snapshots(run_dir, registry_path) == snapshots
    assert not temp_alias.exists()
    assert not temp_alias.with_suffix(temp_alias.suffix + ".lock").exists()


def test_add_experiment_rejects_run_artifact_in_managed_atomic_temp_namespace(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "run_artifact_temp_alias")
    temp_alias = run_dir / ".artifact_hashes.json.collision.tmp"
    temp_alias.write_bytes(b"must remain untouched\n")
    snapshots = _transaction_snapshots(run_dir, registry_path)

    result = add_experiment(run_dir, registry_path=registry_path)

    assert "error" in result and "overlap" in result["error"]
    assert _transaction_snapshots(run_dir, registry_path) == snapshots
    assert temp_alias.read_bytes() == b"must remain untouched\n"


def test_add_experiment_holds_final_lock_through_registry_append(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    run_dir, registry_path = _write_transactional_run(
        workspace,
        "final_lock_race",
        run_relative=Path("versions/v001/09_backtests/run_1"),
    )
    config_path = workspace / ".open-xquant/workspace.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("workflow:\n  layout: version_governed\n", encoding="utf-8")
    append_entered = threading.Event()
    allow_append = threading.Event()
    original_append = experiment_registry._append_registry_entry_locked
    result: list[dict] = []

    def paused_append(path, entry):
        append_entered.set()
        assert allow_append.wait(timeout=5)
        return original_append(path, entry)

    monkeypatch.setattr(experiment_registry, "_append_registry_entry_locked", paused_append)
    worker = threading.Thread(
        target=lambda: result.append(add_experiment(run_dir, registry_path=registry_path)),
    )
    worker.start()
    assert append_entered.wait(timeout=5)
    marker = tmp_path / "final-lock-acquired"
    script = """
import sys
from pathlib import Path
from oxq.selection_lock import final_selection_lock_path, hold_final_selection_lock

run_dir = Path(sys.argv[1])
marker = Path(sys.argv[2])
with hold_final_selection_lock(final_selection_lock_path(run_dir)):
    marker.write_text("acquired\\n", encoding="utf-8")
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(run_dir), str(marker)],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        time.sleep(0.15)
        assert process.poll() is None
        assert not marker.exists()
    finally:
        allow_append.set()
    worker.join(timeout=5)
    stdout, stderr = process.communicate(timeout=5)

    assert not worker.is_alive()
    assert process.returncode == 0, (stdout, stderr)
    assert marker.exists()
    assert len(result) == 1 and "error" not in result[0]


def test_add_experiment_recovers_crashed_registry_transaction_before_next_audit(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "crash_recovery")
    snapshots = _transaction_snapshots(run_dir, registry_path)
    script = """
import os
import sys
from pathlib import Path
from oxq.observe import experiment_registry

run_dir = Path(sys.argv[1])
registry_path = Path(sys.argv[2])
experiment_registry._append_registry_entry_locked = lambda *_args, **_kwargs: os._exit(79)
experiment_registry.add_experiment(run_dir, registry_path=registry_path)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), str(registry_path)],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        check=False,
    )
    assert process.returncode == 79, (process.stdout, process.stderr)
    journal_path = registry_path.with_suffix(registry_path.suffix + ".transaction.json")
    assert journal_path.is_file()
    assert _transaction_snapshots(run_dir, registry_path) != snapshots

    def stop_after_recovery(_run_path):
        raise RuntimeError("stop after recovery")

    monkeypatch.setattr("oxq.audit.research_bias.audit_research", stop_after_recovery)
    with pytest.raises(RuntimeError, match="stop after recovery"):
        add_experiment(run_dir, registry_path=registry_path)

    assert _transaction_snapshots(run_dir, registry_path) == snapshots
    assert not journal_path.exists()


@pytest.mark.parametrize("rollback_replacement", [1, 2, 3, 4])
def test_digest_readers_resume_registry_rollback_after_each_replacement(
    tmp_path,
    rollback_replacement: int,
) -> None:
    run_dir, registry_path = _write_transactional_run(
        tmp_path,
        f"digest_first_rollback_{rollback_replacement}",
    )
    snapshots = _transaction_snapshots(run_dir, registry_path)
    script = """
import os
import sys
from pathlib import Path
from oxq.observe import experiment_registry

run_dir = Path(sys.argv[1])
registry_path = Path(sys.argv[2])
stop_after = int(sys.argv[3])
rollback_targets = {
    run_dir / "research_bias_audit.json",
    run_dir / "artifact_hashes.json",
    run_dir.parent / "run_digests.jsonl",
    registry_path,
}
original_replace = os.replace
rolling_back = False
replacement_count = 0

def fail_append(*_args, **_kwargs):
    global rolling_back
    rolling_back = True
    raise OSError("start rollback")

def exit_after_selected_replacement(source, target):
    global replacement_count
    original_replace(source, target)
    if rolling_back and Path(target) in rollback_targets:
        replacement_count += 1
        if replacement_count == stop_after:
            os._exit(90 + stop_after)

experiment_registry._append_registry_entry_locked = fail_append
os.replace = exit_after_selected_replacement
experiment_registry.add_experiment(run_dir, registry_path=registry_path)
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(run_dir),
            str(registry_path),
            str(rollback_replacement),
        ],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        check=False,
    )
    assert process.returncode == 90 + rollback_replacement, (process.stdout, process.stderr)

    require_current_run_digest(run_dir)
    with multi_run_digest_read_transaction([run_dir]) as canonical:
        assert canonical == (run_dir,)

    assert _transaction_snapshots(run_dir, registry_path)[:3] == snapshots[:3]
    assert experiment_registry.list_experiments(registry_path) == [
        {"experiment_id": "existing"}
    ]
    assert _transaction_snapshots(run_dir, registry_path) == snapshots
    assert not (run_dir.parent / "run_digests.jsonl.journal").exists()
    assert not registry_path.with_suffix(
        registry_path.suffix + ".transaction.json"
    ).exists()


def test_list_experiments_returns_empty_without_creating_missing_parent(tmp_path) -> None:
    missing_parent = tmp_path / "missing" / "registry"
    registry_path = missing_parent / "experiments.jsonl"

    assert experiment_registry.list_experiments(registry_path) == []
    assert not (tmp_path / "missing").exists()


def test_list_experiments_waits_for_writer_rollback_and_returns_recovered_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "locked_reader_rollback")
    append_completed = threading.Event()
    allow_writer_failure = threading.Event()
    reader_completed = threading.Event()
    original_append = experiment_registry._append_registry_entry_locked
    writer_results: list[dict] = []
    reader_results: list[list[dict]] = []
    reader_failures: list[BaseException] = []

    def append_then_fail(path: Path, entry: dict) -> None:
        original_append(path, entry)
        append_completed.set()
        assert allow_writer_failure.wait(timeout=5)
        raise OSError("injected post-append failure")

    def read_registry() -> None:
        try:
            reader_results.append(experiment_registry.list_experiments(registry_path))
        except BaseException as exc:
            reader_failures.append(exc)
        finally:
            reader_completed.set()

    monkeypatch.setattr(experiment_registry, "_append_registry_entry_locked", append_then_fail)
    writer = threading.Thread(
        target=lambda: writer_results.append(add_experiment(run_dir, registry_path=registry_path)),
        name="registry-writer",
    )
    writer.start()
    assert append_completed.wait(timeout=5)
    reader = threading.Thread(target=read_registry, name="registry-reader")
    reader.start()
    try:
        reader_blocked = not reader_completed.wait(timeout=0.2)
    finally:
        allow_writer_failure.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert reader_blocked
    assert not writer.is_alive()
    assert not reader.is_alive()
    assert writer_results == [
        {"error": "experiment registry append failed: injected post-append failure"}
    ]
    assert reader_failures == []
    assert reader_results == [[{"experiment_id": "existing"}]]


def test_list_experiments_recovers_crash_after_append_before_journal_clear(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "reader_crash_recovery")
    script = """
import os
import sys
from pathlib import Path
from oxq.observe import experiment_registry

run_dir = Path(sys.argv[1])
registry_path = Path(sys.argv[2])
experiment_registry._clear_transaction_journal = lambda *_args, **_kwargs: os._exit(81)
experiment_registry.add_experiment(run_dir, registry_path=registry_path)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), str(registry_path)],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        check=False,
    )
    journal_path = registry_path.with_suffix(registry_path.suffix + ".transaction.json")
    assert process.returncode == 81, (process.stdout, process.stderr)
    assert journal_path.is_file()

    entries = experiment_registry.list_experiments(registry_path)

    assert entries == [{"experiment_id": "existing"}]
    assert not journal_path.exists()
    require_current_run_digest(run_dir)


def test_list_experiments_recovers_registry_journal_through_alternate_case_alias(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(tmp_path, "alternate_case_registry")
    actual_registry = registry_path.with_name("Experiments.JSONL")
    registry_path.replace(actual_registry)
    alias_registry = _case_insensitive_alias(actual_registry)
    script = """
import os
import sys
from pathlib import Path
from oxq.observe import experiment_registry

run_dir = Path(sys.argv[1])
registry_path = Path(sys.argv[2])
experiment_registry._clear_transaction_journal = lambda *_args, **_kwargs: os._exit(82)
experiment_registry.add_experiment(run_dir, registry_path=registry_path)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(run_dir), str(actual_registry)],
        cwd=Path.cwd(),
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        check=False,
    )
    journal_path = actual_registry.with_suffix(actual_registry.suffix + ".transaction.json")
    assert process.returncode == 82, (process.stdout, process.stderr)
    assert journal_path.is_file()

    entries = experiment_registry.list_experiments(alias_registry)

    assert entries == [{"experiment_id": "existing"}]
    assert not journal_path.exists()
    require_current_run_digest(run_dir)


def test_add_experiment_uses_actual_run_entry_spelling_through_case_alias(tmp_path) -> None:
    run_dir, registry_path = _write_transactional_run(
        tmp_path,
        "actual_case_run_id",
        run_relative=Path("ActualRun"),
    )
    alias_run = _case_insensitive_alias(run_dir)

    result = add_experiment(alias_run, registry_path=registry_path)

    assert "error" not in result
    assert result["run_id"] == "ActualRun"
    require_current_run_digest(run_dir)


def test_add_experiment_records_version_governed_lineage_fields(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    run_id = "20260709_060149_demo"
    run_dir = workspace / "versions" / "v003" / "09_backtests" / run_id
    run_dir.mkdir(parents=True)
    registry_path = workspace / "experiments.jsonl"
    monkeypatch.chdir(workspace)

    spec = StrategySpec.template(strategy_id="version_registry", hypothesis="lineage fields are preserved")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "version_registry", "run_id": run_id, "trade_count": 12}),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash(), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert entry["version_id"] == "v003"
    assert entry["run_path"] == f"versions/v003/09_backtests/{run_id}"
    assert entry["run_role"] == "primary"


def test_add_experiment_infers_version_from_backtest_layout_not_parent_name(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "versions" / "workspace"
    run_id = "20260709_060149_demo"
    run_dir = workspace / "versions" / "v003" / "09_backtests" / run_id
    run_dir.mkdir(parents=True)
    registry_path = workspace / "experiments.jsonl"
    monkeypatch.chdir(workspace)

    spec = StrategySpec.template(strategy_id="nested_versions_parent", hypothesis="version inference uses run layout")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "nested_versions_parent", "run_id": run_id, "trade_count": 12}),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash(), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert entry["version_id"] == "v003"
    assert entry["run_path"] == f"versions/v003/09_backtests/{run_id}"


def test_add_experiment_infers_version_from_configured_version_root(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "research_versions" / "v003" / "09_backtests" / "run_1"
    run_dir.mkdir(parents=True)
    registry_path = workspace / "experiments.jsonl"
    monkeypatch.chdir(workspace)
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "custom_root", "run_id": "run_1"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(run_dir, "custom_root")

    entry = add_experiment(
        run_dir,
        registry_path=registry_path,
        version_root=workspace / "research_versions",
    )

    assert entry["version_id"] == "v003"


def test_configured_version_root_does_not_infer_lineage_below_run_depth(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    version_root = workspace / "research_versions"
    payload_dir = version_root / "v003" / "09_backtests" / "run_1" / "payload"
    payload_dir.mkdir(parents=True)
    registry_path = workspace / "experiments.jsonl"
    monkeypatch.chdir(workspace)
    (payload_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "nested_payload", "run_id": "payload"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(payload_dir, "nested_payload")

    entry = add_experiment(
        payload_dir,
        registry_path=registry_path,
        version_root=version_root,
    )

    assert entry["version_id"] == ""
    assert entry["run_role"] == "primary"


def test_add_experiment_uses_explicit_manifest_backtest_phase(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    backtest_phase_dir = workspace / "versions/v003/artifacts/backtests"
    run_dir = backtest_phase_dir / "run_1_cost_x2"
    run_dir.mkdir(parents=True)
    monkeypatch.chdir(workspace)
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "custom_phase", "run_id": "run_1_cost_x2"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(run_dir, "custom_phase")

    entry = add_experiment(
        run_dir,
        registry_path=workspace / "experiments.jsonl",
        backtest_phase_dir=backtest_phase_dir,
        version_id="v003",
    )

    assert entry["version_id"] == "v003"
    assert entry["run_role"] == "robustness_cost_x2"


def test_add_experiment_rejects_run_outside_explicit_backtest_phase(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    backtest_phase_dir = workspace / "versions/v003/artifacts/backtests"
    run_dir = workspace / "versions/v003/elsewhere/run_1"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "outside_phase", "run_id": "run_1"}),
        encoding="utf-8",
    )
    registry_path = workspace / "experiments.jsonl"

    entry = add_experiment(
        run_dir,
        registry_path=registry_path,
        backtest_phase_dir=backtest_phase_dir,
        version_id="v003",
    )

    assert entry == {"error": "run directory must stay within the resolved backtest phase directory"}
    assert not registry_path.exists()


def test_add_experiment_uses_structural_backtest_under_custom_root_named_backtests(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    version_root = workspace / "archive" / "09_backtests"
    run_dir = version_root / "v003" / "09_backtests" / "run_1_cost_x2"
    run_dir.mkdir(parents=True)
    registry_path = workspace / "experiments.jsonl"
    monkeypatch.chdir(workspace)
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "custom_nested_root", "run_id": "run_1_cost_x2"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(run_dir, "custom_nested_root")

    entry = add_experiment(run_dir, registry_path=registry_path, version_root=version_root)

    assert entry["version_id"] == "v003"
    assert entry["run_role"] == "robustness_cost_x2"


def test_explicit_version_root_rejects_outside_structural_version_path(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    version_root = workspace / "versions"
    run_dir = tmp_path / "outside" / "v999" / "09_backtests" / "run_1_cost_x2"
    run_dir.mkdir(parents=True)
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "outside_root", "run_id": "run_1_cost_x2"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(run_dir, "outside_root")

    entry = add_experiment(
        run_dir,
        registry_path=workspace / "experiments.jsonl",
        version_root=version_root,
    )

    assert entry["version_id"] == ""
    assert entry["run_role"] == "primary"


def test_add_experiment_marks_cost_x2_version_governed_run(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    run_id = "20260709_060149_demo_cost_x2"
    run_dir = workspace / "versions" / "v003" / "09_backtests" / run_id
    run_dir.mkdir(parents=True)
    registry_path = workspace / "experiments.jsonl"
    monkeypatch.chdir(workspace)

    spec = StrategySpec.template(strategy_id="cost_x2_registry", hypothesis="robustness role is preserved")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "cost_x2_registry", "run_id": run_id, "trade_count": 12}),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash(), encoding="utf-8")
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert entry["version_id"] == "v003"
    assert entry["run_role"] == "robustness_cost_x2"


def test_add_experiment_ids_are_unique_within_same_second(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"

    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "audit_registry", "run_id": "run_1"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(run_dir, "audit_registry")
    (run_dir / "research_bias_audit.json").write_text(
        json.dumps({"status": "pass"}),
        encoding="utf-8",
    )

    first = add_experiment(run_dir, registry_path=registry_path)
    second = add_experiment(run_dir, registry_path=registry_path)

    assert first["experiment_id"] != second["experiment_id"]


def test_add_experiment_returns_error_for_corrupt_metrics_json(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    (run_dir / "metrics.json").write_text("{not-json", encoding="utf-8")

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert "error" in entry
    assert not registry_path.exists()


def test_add_experiment_returns_error_for_non_object_metrics_json(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    (run_dir / "metrics.json").write_text("[]", encoding="utf-8")

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert "error" in entry
    assert not registry_path.exists()


def test_add_experiment_rejects_metrics_run_id_mismatch_before_mutation(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "identity_mismatch", "run_id": "different_run"}),
        encoding="utf-8",
    )

    entry = add_experiment(run_dir, registry_path=registry_path)

    assert entry == {"error": "metrics.json run_id must match the resolved run directory name: run_1"}
    assert not registry_path.exists()
    assert not (run_dir / "research_bias_audit.json").exists()


def test_structural_fallback_does_not_assign_parent_run_to_nested_payload(
    tmp_path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    payload_dir = workspace / "archive/v009/09_backtests/run_1/payload"
    payload_dir.mkdir(parents=True)
    monkeypatch.chdir(workspace)
    (payload_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "nested_payload", "run_id": "payload"}),
        encoding="utf-8",
    )
    _write_identity_artifacts(payload_dir, "nested_payload")

    entry = add_experiment(payload_dir, registry_path=workspace / "experiments.jsonl")

    assert entry["version_id"] == ""
    assert entry["run_role"] == "primary"


def test_add_experiment_reruns_stale_research_audit(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    registry_path = tmp_path / "experiments.jsonl"

    spec = StrategySpec.template(strategy_id="stale_audit", hypothesis="stale audits are not trusted")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.dump(spec.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"strategy_id": "stale_audit", "run_id": "run_1", "trade_count": 12}),
        encoding="utf-8",
    )
    (run_dir / "data_manifest.json").write_text(
        json.dumps({"symbols": ["SPY"], "missing_ratio": 0.0}),
        encoding="utf-8",
    )
    (run_dir / "research_bias_audit.json").write_text(
        json.dumps({"status": "pass"}),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")

    entry = add_experiment(run_dir, registry_path=registry_path)

    audit = json.loads((run_dir / "research_bias_audit.json").read_text(encoding="utf-8"))
    assert entry["audit_status"] == "fail"
    assert audit["status"] == "fail"


def _file_hash(path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()[:16]}"


def _write_transactional_run(
    root,
    strategy_id: str,
    *,
    run_relative: Path = Path("run_1"),
) -> tuple[Path, Path]:
    run_dir = root / run_relative
    run_dir.mkdir(parents=True)
    spec = StrategySpec.template(strategy_id=strategy_id, hypothesis="registry transaction fixture")
    _write_artifacts(
        spec,
        RunResult(
            portfolio=Portfolio(cash=Decimal("100000")),
            trades=[],
            equity_curve=[("2024-01-02", 100_000.0), ("2024-01-03", 90_000.0)],
            mktdata={},
        ),
        run_dir,
        Engine(),
    )
    old_audit = b'{"status":"old","generation":1}\n'
    (run_dir / "research_bias_audit.json").write_bytes(old_audit)
    update_artifact_hashes_and_run_digest(
        run_dir,
        lambda hashes: hashes.__setitem__("research_bias_audit.json", _file_hash(run_dir / "research_bias_audit.json")),
    )
    registry_path = root / "experiments.jsonl"
    registry_path.write_bytes(b'{"experiment_id":"existing"}\n')
    return run_dir, registry_path


def _transaction_snapshots(
    run_dir: Path,
    registry_path: Path,
    *,
    follow_registry: bool = True,
) -> tuple[bytes, bytes, bytes, bytes]:
    registry = registry_path.read_bytes() if follow_registry else b""
    return (
        (run_dir / "research_bias_audit.json").read_bytes(),
        (run_dir / "artifact_hashes.json").read_bytes(),
        (run_dir.parent / "run_digests.jsonl").read_bytes(),
        registry,
    )


def _write_identity_artifacts(run_dir, strategy_id: str) -> None:
    spec = StrategySpec.template(strategy_id=strategy_id, hypothesis="experiment identity fixture")
    (run_dir / "strategy_spec.yaml").write_text(
        yaml.safe_dump(spec.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "spec_hash.txt").write_text(spec.compute_hash() + "\n", encoding="utf-8")
