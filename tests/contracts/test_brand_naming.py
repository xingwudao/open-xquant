import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
CANONICAL_BRANDS = (
    (re.compile(r"\bopen(?:-| )?xquant\b", re.IGNORECASE), "open-xquant"),
    (re.compile(r"\bequant(?:-| )py\b", re.IGNORECASE), "equant-py"),
)


def _tracked_text_files() -> list[tuple[Path, str]]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    tracked: list[tuple[Path, str]] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = ROOT / os.fsdecode(raw_path)
        content = path.read_bytes()
        if b"\0" in content:
            continue
        try:
            tracked.append((path, content.decode("utf-8")))
        except UnicodeDecodeError:
            continue
    return tracked


def test_tracked_text_uses_canonical_repository_brand_names() -> None:
    violations: list[str] = []
    for path, content in _tracked_text_files():
        for pattern, canonical in CANONICAL_BRANDS:
            violations.extend(
                f"{path.relative_to(ROOT)}: {match.group(0)!r} != {canonical!r}"
                for match in pattern.finditer(content)
                if match.group(0) != canonical
            )
    assert violations == []
