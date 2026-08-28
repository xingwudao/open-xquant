"""Release-index parsing and target compatibility tests."""

from __future__ import annotations

from copy import deepcopy

import pytest
from packaging.tags import Tag

from oxq.operators.install_errors import OperatorInstallError
from oxq.operators.release_index import (
    load_official_provider,
    parse_exact_requirement,
    parse_release_index,
    select_release_target,
)


def _release_index() -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    asset = {
        "filename": "equant-py-1.0.0.open-xquant-certification.zip",
        "url": "https://github.com/xingwudao/equant-py/releases/download/v1.0.0/bundle.zip",
        "size_bytes": 1,
        "digest": digest,
    }
    return {
        "schema_version": 1,
        "release_type": "open-xquant-operator-release",
        "provider": "equant-py",
        "release": "1.0.0",
        "submission_commit": "git-sha1:" + "a" * 40,
        "source_commit": "git-sha1:" + "b" * 40,
        "certification_state": "research-certified",
        "operator_count": 1,
        "targets": [
            {
                "python_tag": "cp312",
                "abi_tag": "cp312",
                "platform_tag": "macosx_14_0_arm64",
                "bundle": asset,
                "wheels": [
                    {
                        **asset,
                        "filename": "equant_ttr-1.0.0-py3-none-any.whl",
                        "distribution": "equant-ttr",
                        "version": "1.0.0",
                        "role": "implementation",
                        "tags": ["py3-none-any"],
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "value",
    [
        "equant-py",
        "equant-py>=1.0.0",
        "equant-py==1.0",
        "equant-py==1!1.0.0",
        "equant-py==1.0.0+local",
    ],
)
def test_requirement_rejects_non_exact_semver(value: str) -> None:
    with pytest.raises(OperatorInstallError) as caught:
        parse_exact_requirement(value)
    assert caught.value.code == "operator_requirement_invalid"


def test_requirement_returns_canonical_provider_and_semver() -> None:
    assert parse_exact_requirement("equant-py==1.2.3-rc.1") == ("equant-py", "1.2.3-rc.1")


def test_loads_frozen_official_provider() -> None:
    provider = load_official_provider("equant-py")

    assert provider.name == "equant-py"
    assert provider.repository == "xingwudao/equant-py"
    assert provider.release_asset == "operator-release-v1.json"


def test_rejects_unknown_official_provider() -> None:
    with pytest.raises(OperatorInstallError) as caught:
        load_official_provider("unknown-provider")
    assert caught.value.code == "operator_provider_unknown"
    assert caught.value.as_dict()["provider"] == "unknown-provider"


def test_install_error_dict_removes_url_credentials_and_query_strings() -> None:
    error = OperatorInstallError(
        "operator_download_failed",
        "download failed: https://user:secret@example.test/file?token=secret",
        stage="download",
        provider="equant-py",
    )

    serialized = error.as_dict()

    assert serialized["message"] == "download failed: https://example.test/file"
    assert "secret" not in repr(serialized)


def test_parses_strict_canonical_release_index() -> None:
    parsed = parse_release_index(
        b'{"certification_state":"research-certified","operator_count":1,"provider":"equant-py","release":"1.0.0","release_type":"open-xquant-operator-release","schema_version":1,"source_commit":"git-sha1:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","submission_commit":"git-sha1:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","targets":[{"abi_tag":"cp312","bundle":{"digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","filename":"equant-py-1.0.0.open-xquant-certification.zip","size_bytes":1,"url":"https://github.com/xingwudao/equant-py/releases/download/v1.0.0/bundle.zip"},"platform_tag":"macosx_14_0_arm64","python_tag":"cp312","wheels":[{"digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","distribution":"equant-ttr","filename":"equant_ttr-1.0.0-py3-none-any.whl","role":"implementation","size_bytes":1,"tags":["py3-none-any"],"url":"https://github.com/xingwudao/equant-py/releases/download/v1.0.0/bundle.zip","version":"1.0.0"}]}]}\n'
    )

    assert parsed.provider == "equant-py"
    assert parsed.targets[0].wheels[0].tags == ("py3-none-any",)


@pytest.mark.parametrize("mutation", ["extra_field", "too_many_targets", "too_many_wheels"])
def test_rejects_schema_extra_fields_and_collection_bounds(mutation: str) -> None:
    release = _release_index()
    if mutation == "extra_field":
        release["unexpected"] = True
    elif mutation == "too_many_targets":
        release["targets"] = [deepcopy(release["targets"][0]) for _ in range(17)]  # type: ignore[index]
    else:
        target = release["targets"][0]  # type: ignore[index]
        target["wheels"] = [deepcopy(target["wheels"][0]) for _ in range(129)]  # type: ignore[index]

    with pytest.raises(OperatorInstallError) as caught:
        parse_release_index(_canonical_bytes(release))
    assert caught.value.code == "operator_release_invalid"


def test_rejects_noncanonical_release_index_bytes() -> None:
    with pytest.raises(OperatorInstallError) as caught:
        parse_release_index(b'{"schema_version": 1}\n')
    assert caught.value.code == "operator_release_invalid"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("provider",), "equant-py\n"),
        (("release",), "1.0.0\n"),
        (("targets", 0, "python_tag"), "cp312\n"),
        (("targets", 0, "wheels", 0, "distribution"), "equant-ttr\n"),
        (("targets", 0, "wheels", 0, "tags", 0), "py3-none-any\n"),
    ],
)
def test_rejects_newline_suffixed_schema_constrained_identifiers(
    path: tuple[str | int, ...],
    value: str,
) -> None:
    release = _release_index()
    _set_nested_value(release, path, value)

    with pytest.raises(OperatorInstallError) as caught:
        parse_release_index(_canonical_bytes(release))
    assert caught.value.code == "operator_release_invalid"


def test_selects_the_single_compatible_target(monkeypatch: pytest.MonkeyPatch) -> None:
    index = parse_release_index(_canonical_bytes(_release_index()))
    monkeypatch.setattr(
        "oxq.operators.release_index.tags.sys_tags",
        lambda: iter((Tag("cp312", "cp312", "macosx_14_0_arm64"), Tag("py3", "none", "any"))),
    )

    assert select_release_target(index) == index.targets[0]


def test_rejects_zero_compatible_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    index = parse_release_index(_canonical_bytes(_release_index()))
    monkeypatch.setattr(
        "oxq.operators.release_index.tags.sys_tags",
        lambda: iter((Tag("cp311", "cp311", "manylinux_2_17_x86_64"),)),
    )

    with pytest.raises(OperatorInstallError) as caught:
        select_release_target(index)
    assert caught.value.code == "operator_target_unavailable"


def test_rejects_multiple_compatible_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    release = _release_index()
    release["targets"] = [deepcopy(release["targets"][0]) for _ in range(2)]  # type: ignore[index]
    index = parse_release_index(_canonical_bytes(release))
    monkeypatch.setattr(
        "oxq.operators.release_index.tags.sys_tags",
        lambda: iter((Tag("cp312", "cp312", "macosx_14_0_arm64"), Tag("py3", "none", "any"))),
    )

    with pytest.raises(OperatorInstallError) as caught:
        select_release_target(index)
    assert caught.value.code == "operator_release_invalid"


def test_rejects_compatible_target_with_current_tag_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    index = parse_release_index(_canonical_bytes(_release_index()))
    monkeypatch.setattr(
        "oxq.operators.release_index.tags.sys_tags",
        lambda: iter((Tag("cp312", "cp312", "macosx_14_0_arm64"),)),
    )

    with pytest.raises(OperatorInstallError) as caught:
        select_release_target(index)
    assert caught.value.code == "operator_release_invalid"


def test_accepts_target_when_each_wheel_has_any_compatible_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _release_index()
    wheel = release["targets"][0]["wheels"][0]  # type: ignore[index]
    wheel["tags"] = ["py3-none-any", "cp311-cp311-manylinux_2_17_x86_64"]
    index = parse_release_index(_canonical_bytes(release))
    monkeypatch.setattr(
        "oxq.operators.release_index.tags.sys_tags",
        lambda: iter((Tag("cp312", "cp312", "macosx_14_0_arm64"), Tag("py3", "none", "any"))),
    )

    assert select_release_target(index) == index.targets[0]


def test_rejects_target_when_a_wheel_has_no_compatible_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _release_index()
    wheel = release["targets"][0]["wheels"][0]  # type: ignore[index]
    wheel["tags"] = ["cp311-cp311-manylinux_2_17_x86_64"]
    index = parse_release_index(_canonical_bytes(release))
    monkeypatch.setattr(
        "oxq.operators.release_index.tags.sys_tags",
        lambda: iter((Tag("cp312", "cp312", "macosx_14_0_arm64"), Tag("py3", "none", "any"))),
    )

    with pytest.raises(OperatorInstallError) as caught:
        select_release_target(index)
    assert caught.value.code == "operator_release_invalid"


def _canonical_bytes(value: dict[str, object]) -> bytes:
    import json

    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _set_nested_value(
    value: dict[str, object],
    path: tuple[str | int, ...],
    replacement: str,
) -> None:
    current: object = value
    for part in path[:-1]:
        current = current[part]  # type: ignore[index]
    current[path[-1]] = replacement  # type: ignore[index]
