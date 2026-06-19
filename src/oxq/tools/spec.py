"""Spec tools — strategy spec creation and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from oxq.spec.schema import StrategySpec, make_strategy_id
from oxq.spec.validator import validate as validate_spec
from oxq.tools.registry import registry


@registry.tool(
    name="spec_init",
    description="Initialize a new strategy spec from a natural language description. "
    "Creates a strategy_spec.yaml template with required fields pre-filled.",
)
def spec_init(description: str, out: str = "strategy_spec.yaml") -> dict[str, Any]:
    """Create a strategy spec template."""
    strategy_id = make_strategy_id(description)
    template = StrategySpec.template(strategy_id=strategy_id, hypothesis=description)

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(template.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "strategy_id": strategy_id,
        "output": str(output_path),
        "next": "Edit the file, then call spec_validate",
    }


@registry.tool(
    name="spec_validate",
    description="Validate a strategy spec file against P0 rules. "
    "Checks hypothesis, universe, signal timing, execution semantics, supported calendar, "
    "costs, OOS period, metrics profile, and benchmark. "
    "Returns status (pass/fail), errors, warnings, and spec_hash.",
)
def spec_validate(spec_file: str) -> dict[str, Any]:
    """Validate a strategy spec file."""
    try:
        parsed = StrategySpec.from_yaml(spec_file)
    except Exception as e:
        return {
            "status": "fail",
            "errors": [{"severity": "fatal", "check": "parse_error", "message": str(e)}],
            "warnings": [],
            "spec_hash": "",
        }

    result = validate_spec(parsed)
    return result.to_dict()
