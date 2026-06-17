from __future__ import annotations

import json

import pytest

from oxq.cli.agent_manifest import (
    MarkerBlockError,
    expand_path,
    read_json_file,
    remove_marker_block,
    sha256_file,
    upsert_marker_block,
    write_json_file,
)


def test_expand_path_uses_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert expand_path("~/agent.json") == tmp_path / "agent.json"


def test_json_round_trip_and_sha(tmp_path) -> None:
    path = tmp_path / "nested" / "manifest.json"

    write_json_file(path, {"schema_version": 1, "name": "open-xquant"})

    assert read_json_file(path)["name"] == "open-xquant"
    assert sha256_file(path) == sha256_file(path)
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_marker_block_insert_replace_and_remove(tmp_path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text("before\n", encoding="utf-8")

    upsert_marker_block(path, "open-xquant", "managed v1")
    upsert_marker_block(path, "open-xquant", "managed v2")

    text = path.read_text(encoding="utf-8")
    assert "before" in text
    assert "managed v1" not in text
    assert "managed v2" in text
    assert text.count("open-xquant:begin") == 1

    remove_marker_block(path, "open-xquant")

    assert path.read_text(encoding="utf-8") == "before\n"


def test_marker_block_partial_marker_fails(tmp_path) -> None:
    path = tmp_path / "AGENTS.md"
    path.write_text("before\n<!-- open-xquant:begin -->\nmissing end\n", encoding="utf-8")

    with pytest.raises(MarkerBlockError):
        upsert_marker_block(path, "open-xquant", "managed")
