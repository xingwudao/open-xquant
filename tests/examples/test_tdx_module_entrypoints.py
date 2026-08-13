from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "module_name",
    (
        "examples.modules.pytdx_downloader",
        "examples.modules.tdxquant_downloader",
    ),
)
def test_tdx_downloader_is_importable_from_modules(module_name: str) -> None:
    assert importlib.util.find_spec(module_name) is not None


@pytest.mark.parametrize(
    "script_name",
    ("pytdx_downloader.py", "tdxquant_downloader.py"),
)
def test_tdx_downloader_help_runs_from_modules(script_name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "modules" / script_name), "--help"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
