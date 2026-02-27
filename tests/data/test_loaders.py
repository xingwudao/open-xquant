from pathlib import Path

from oxq.data.loaders import resolve_data_dir


def test_resolve_with_explicit_dir(tmp_path: Path) -> None:
    result = resolve_data_dir(tmp_path / "custom")
    assert result == tmp_path / "custom"


def test_resolve_with_env_var(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path / "env"))
    result = resolve_data_dir()
    assert result == tmp_path / "env" / "market"


def test_resolve_default(monkeypatch) -> None:
    monkeypatch.delenv("OXQ_DATA_DIR", raising=False)
    result = resolve_data_dir()
    assert result == Path.home() / ".oxq" / "data" / "market"


def test_explicit_dir_overrides_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OXQ_DATA_DIR", str(tmp_path / "env"))
    result = resolve_data_dir(tmp_path / "explicit")
    assert result == tmp_path / "explicit"
