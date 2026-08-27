"""Exact, bounded wheel-closure verification."""

from __future__ import annotations

import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from oxq.operators.install_errors import OperatorInstallError
from oxq.operators.wheel_closure import verify_wheel_closure
from tests.operators.wheel_helpers import (
    encrypted_info,
    fifo_info,
    symlink_info,
    target,
    wheel_record,
)


def _verify(tmp_path: Path, wheels: list[object], *, certified: object | None = None):
    artifacts = wheels if certified is None else certified
    return verify_wheel_closure(
        target(*wheels),
        [tmp_path / wheel.filename for wheel in wheels],
        certified_artifacts=artifacts,
    )


def test_verifies_exact_wheel_closure_and_identity(tmp_path: Path) -> None:
    core = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", requires=("equant-ttr==1.0.0",))
    ttr = wheel_record(tmp_path / "equant_ttr-1.0.0-py3-none-any.whl", distribution="equant-ttr")

    result = _verify(tmp_path, [core, ttr])

    assert [wheel.distribution for wheel in result.wheels] == ["equant-core", "equant-ttr"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("distribution", "other"),
        ("version", "2.0.0"),
        ("tag", "cp311-cp311-manylinux_2_17_x86_64"),
    ],
)
def test_rejects_release_identity_mismatch(tmp_path: Path, field: str, value: str) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", **{field: value})
    with pytest.raises(OperatorInstallError):
        verify_wheel_closure(
            target(replace(wheel, distribution="equant-core", version="1.0.0", tags=("py3-none-any",))),
            [tmp_path / wheel.filename],
            certified_artifacts=[wheel],
        )


def test_rejects_unresolved_dependency(tmp_path: Path) -> None:
    requires = ("missing==1.0.0",)
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", requires=requires)
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


def test_rejects_direct_url_dependency_even_when_named_wheel_is_present(tmp_path: Path) -> None:
    wheel = wheel_record(
        tmp_path / "equant_core-1.0.0-py3-none-any.whl",
        requires=("equant-core @ https://example.invalid/equant-core.whl",),
    )
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


def test_activates_extra_context_for_transitive_dependencies(tmp_path: Path) -> None:
    core = wheel_record(
        tmp_path / "equant_core-1.0.0-py3-none-any.whl",
        requires=("equant-ttr[needed]==1.0.0",),
    )
    ttr = wheel_record(
        tmp_path / "equant_ttr-1.0.0-py3-none-any.whl",
        distribution="equant-ttr",
        requires=("missing==1.0.0; extra == 'needed'",),
    )
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [core, ttr])


@pytest.mark.parametrize(
    "argument, value",
    [
        ("metadata_text", "Metadata-Version: 2.1\nName: equant-core\n"),
        ("wheel_text", "Wheel-Version: 2.0\nTag: py3-none-any\n"),
    ],
)
def test_rejects_invalid_metadata_and_wheel_headers(tmp_path: Path, argument: str, value: str) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", **{argument: value})
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


@pytest.mark.parametrize(
    "wheel_text",
    [
        "Wheel-Version: 1.0\nTag: py3-none-any\n",
        "Wheel-Version: 1.0\nWheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        "bad header\nWheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    ],
)
def test_rejects_incomplete_or_defective_wheel_headers(tmp_path: Path, wheel_text: str) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", wheel_text=wheel_text)
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


def test_rejects_extra_downloaded_wheel(tmp_path: Path) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl")
    extra = wheel_record(tmp_path / "equant_extra-1.0.0-py3-none-any.whl", distribution="equant-extra")
    with pytest.raises(OperatorInstallError):
        verify_wheel_closure(target(wheel), [tmp_path / wheel.filename, tmp_path / extra.filename], certified_artifacts=[wheel])


@pytest.mark.parametrize(
    "record",
    [
        "bad\n",
        "equant_core/__init__.py,sha256=wrong,0\n",
        "equant_core/__init__.py,,0\n",
        "equant_core/__init__.py,sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,1\nequant_core-1.0.0.dist-info/RECORD,,\n",
    ],
)
def test_rejects_invalid_record_hash_size_or_completeness(tmp_path: Path, record: str) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", record=record)
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


@pytest.mark.parametrize("entry", ["../escape.py", "/absolute.py", "package\\backslash.py"])
def test_rejects_unsafe_member_paths(tmp_path: Path, entry: str) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", entries={entry: b"x"})
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


@pytest.mark.parametrize("mutation", [symlink_info, encrypted_info, fifo_info])
def test_rejects_symlink_and_encrypted_members(tmp_path: Path, mutation: object) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", mutate_info=mutation)  # type: ignore[arg-type]
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


def test_rejects_duplicate_archive_names(tmp_path: Path) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl")
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(tmp_path / wheel.filename, "a") as archive:
            archive.writestr("equant_core/__init__.py", b"")
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


def test_rejects_expansion_limits(tmp_path: Path) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl", entries={"payload.bin": b"0" * (2 * 1024 * 1024)})
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


def test_rejects_compression_ratio_bound(tmp_path: Path) -> None:
    wheel = wheel_record(
        tmp_path / "equant_core-1.0.0-py3-none-any.whl",
        entries={"payload.bin": b"0" * (512 * 1024)},
    )
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel])


def test_rejects_certification_artifact_mismatch(tmp_path: Path) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl")
    with pytest.raises(OperatorInstallError):
        _verify(tmp_path, [wheel], certified=[])


def test_verifier_never_invokes_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = wheel_record(tmp_path / "equant_core-1.0.0-py3-none-any.whl")
    seen: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: seen.append(args))
    _verify(tmp_path, [wheel])
    assert all("pip" not in repr(args) for args in seen)
