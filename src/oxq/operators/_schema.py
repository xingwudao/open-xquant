"""Access packaged operator contract schemas."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any


def load_contract_schema(name: str) -> dict[str, Any]:
    source_tree_path = Path(__file__).resolve().parents[3] / "contracts" / "quant-operators" / name
    if source_tree_path.is_file():
        raw = source_tree_path.read_text(encoding="utf-8")
    else:
        raw = files("oxq.operators").joinpath("schemas", name).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"operator contract schema must be an object: {name}")
    return payload
