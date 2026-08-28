"""Semantic validation for operator distribution records."""

from __future__ import annotations

from collections.abc import Mapping


def validate_certification_record_v2_semantics(record: Mapping[str, object]) -> None:
    """Validate cross-field invariants that JSON Schema cannot express."""
    baseline_sets = record.get("baseline_sets")
    operators = record.get("operators")
    if not isinstance(baseline_sets, list) or not isinstance(operators, list):
        raise ValueError("certification record structure is invalid")

    baseline_path_counts: dict[str, int] = {}
    for baseline_set in baseline_sets:
        if not isinstance(baseline_set, dict):
            raise ValueError("certification record structure is invalid")
        path = baseline_set.get("path")
        if not isinstance(path, str):
            raise ValueError("certification record structure is invalid")
        baseline_path_counts[path] = baseline_path_counts.get(path, 0) + 1

    for operator in operators:
        if not isinstance(operator, dict):
            raise ValueError("certification record structure is invalid")
        baseline_cases = operator.get("baseline_cases")
        if not isinstance(baseline_cases, list):
            raise ValueError("certification record structure is invalid")
        for baseline_case in baseline_cases:
            if not isinstance(baseline_case, dict):
                raise ValueError("certification record structure is invalid")
            baseline_path = baseline_case.get("baseline_path")
            if not isinstance(baseline_path, str):
                raise ValueError("certification record structure is invalid")
            if baseline_path_counts.get(baseline_path, 0) != 1:
                raise ValueError("every baseline case must bind to exactly one baseline set")
