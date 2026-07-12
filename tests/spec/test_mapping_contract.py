import pytest
from click.testing import CliRunner

from oxq.spec.mapping_contract import validate_mapping_contract, validate_mapping_contract_for_builder_pass
from oxq.spec.schema import IndicatorDef, StrategySpec


def _mapped_strategy_target(target_field: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_format": "external_strategy",
        "source_fields": ["source.dynamic_field"],
        "field_mappings": [
            {
                "source_field": "source.dynamic_field",
                "target_field": target_field,
                "semantic": "strategy",
                "status": "mapped",
                "confirmation_required": False,
                "blocking": False,
                "reason": "Maps a dynamic component field.",
            }
        ],
    }


@pytest.mark.parametrize(
    ("validator", "kwargs"),
    [
        (validate_mapping_contract, {}),
        (validate_mapping_contract_for_builder_pass, {"effective_field_paths": {"portfolio.type"}}),
    ],
)
def test_mapping_contract_rejects_boolean_schema_version(validator, kwargs) -> None:
    payload = _mapped_strategy_target("portfolio.type")
    payload["schema_version"] = True

    result = validator(payload, **kwargs)

    assert result["status"] == "fail"
    assert any(error["path"] == "schema_version" for error in result["errors"])


def test_mapping_contract_accepts_explicit_studio_exclusion() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "target_spec": "versions/v001/04_spec_build/strategy_spec.yaml",
        "field_mappings": [
            {
                "source_field": "output.formats",
                "target_field": "",
                "semantic": "studio",
                "status": "excluded_non_material",
                "confirmation_required": False,
                "blocking": False,
                "reason": "output format belongs to Studio/report configuration, not strategy runtime semantics",
            }
        ],
    }

    result = validate_mapping_contract(payload)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_mapping_contract_rejects_strategy_semantic_without_target_or_block() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "signal.pipeline",
                "target_field": "",
                "semantic": "strategy",
                "status": "excluded_non_material",
                "confirmation_required": False,
                "blocking": False,
                "reason": "not mapped",
            }
        ],
    }

    result = validate_mapping_contract(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_mappings[0].status" for error in result["errors"])


def test_mapping_contract_rejects_blocked_status_without_blocking_flag() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "portfolio.cross_sectional_winsorization",
                "target_field": "",
                "semantic": "strategy",
                "status": "blocked",
                "confirmation_required": False,
                "blocking": False,
                "reason": "No executable target was found for this cross-sectional transform.",
            }
        ],
    }

    result = validate_mapping_contract(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_mappings[0].blocking" for error in result["errors"])


def test_mapping_contract_rejects_unsupported_strategy_semantic_without_blocking() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "rebalance.day",
                "target_field": "",
                "semantic": "strategy",
                "status": "unsupported",
                "confirmation_required": False,
                "blocking": False,
                "reason": "Calendar-aware week-end rebalance is not executable.",
            }
        ],
    }

    result = validate_mapping_contract(payload)

    assert result["status"] == "fail"
    assert any(
        error["path"] == "field_mappings[0].blocking"
        and "unsupported mappings require blocking=true" in error["message"]
        for error in result["errors"]
    )


def test_mapping_contract_rejects_strategy_confirmation_without_blocking() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "market.calendar_mixed_regions",
                "target_field": "market.calendar",
                "semantic": "strategy",
                "status": "needs_user_confirmation",
                "confirmation_required": True,
                "blocking": False,
                "reason": "Single-calendar execution needs user confirmation for a mixed CN+US universe.",
            }
        ],
    }

    result = validate_mapping_contract(payload)

    assert result["status"] == "fail"
    assert any(
        error["path"] == "field_mappings[0].blocking"
        and "strategy semantics needing user confirmation require blocking=true" in error["message"]
        for error in result["errors"]
    )


def test_mapping_contract_builder_pass_rejects_blocked_strategy_row() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "market.cross_calendar_policy",
                "target_field": "",
                "semantic": "strategy",
                "status": "blocked",
                "confirmation_required": False,
                "blocking": True,
                "reason": "Mixed calendar runtime support must be checked before passing the builder gate.",
            }
        ],
    }

    base_result = validate_mapping_contract(payload)
    pass_result = validate_mapping_contract_for_builder_pass(payload)

    assert base_result["status"] == "pass"
    assert pass_result["status"] == "fail"
    assert any(
        error["path"] == "field_mappings[0].status"
        and "builder pass requires mappings to be mapped or excluded_non_material and non-blocking"
        in error["message"]
        for error in pass_result["errors"]
    )


def test_mapping_contract_builder_pass_rejects_unsupported_semantic_bypass() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "portfolio.cross_sectional_winsorization",
                "target_field": "",
                "semantic": "unsupported",
                "status": "unsupported",
                "confirmation_required": False,
                "blocking": False,
                "reason": "No executable target was found for this cross-sectional transform.",
            }
        ],
    }

    base_result = validate_mapping_contract(payload)
    pass_result = validate_mapping_contract_for_builder_pass(payload)

    assert base_result["status"] == "fail"
    assert pass_result["status"] == "fail"
    assert any(error["path"] == "field_mappings[0].blocking" for error in base_result["errors"])
    assert any(error["path"] == "field_mappings[0].status" for error in pass_result["errors"])


def test_mapping_contract_accepts_effective_strategy_target_field() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "backtest.initial_cash",
                "target_field": "execution.initial_cash",
                "semantic": "strategy",
                "status": "mapped",
                "confirmation_required": False,
                "blocking": False,
                "reason": "Initial cash maps to execution assumptions.",
            }
        ],
    }

    result = validate_mapping_contract(payload)

    assert result["status"] == "pass"


def test_mapping_contract_rejects_misplaced_strategy_target_field() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "backtest.initial_cash",
                "target_field": "portfolio.initial_cash",
                "semantic": "strategy",
                "status": "mapped",
                "confirmation_required": False,
                "blocking": False,
                "reason": "This field is intentionally misplaced.",
            }
        ],
    }

    result = validate_mapping_contract(payload)

    assert result["status"] == "fail"
    assert any(
        error["path"] == "field_mappings[0].target_field"
        and "effective StrategySpec field path" in error["message"]
        for error in result["errors"]
    )


def test_mapping_contract_rejects_unknown_dynamic_indicator_structural_field() -> None:
    result = validate_mapping_contract(_mapped_strategy_target("signal.indicators.foo.typo_field"))

    assert result["status"] == "fail"
    assert any(error["path"] == "field_mappings[0].target_field" for error in result["errors"])


@pytest.mark.parametrize(
    "target_field",
    [
        "signal.indicators.roc.type",
        "signal.indicators.roc.lag_bars",
        "signal.indicators.roc.params",
        "signal.indicators.roc.params.period",
        "signal.rules.entry.type",
        "signal.rules.entry.output_domain",
        "signal.rules.entry.params",
        "signal.rules.entry.params.threshold",
        "portfolio.rules.stop_loss.type",
        "portfolio.rules.stop_loss.params",
        "portfolio.rules.stop_loss.params.threshold",
    ],
)
def test_mapping_contract_accepts_structured_dynamic_component_target_fields(target_field: str) -> None:
    result = validate_mapping_contract(_mapped_strategy_target(target_field))

    assert result["status"] == "pass"
    assert result["errors"] == []


@pytest.mark.parametrize(
    "target_field",
    [
        "portfolio.params.nonexistent",
        "portfolio.params.",
        "portfolio.params..n",
    ],
)
def test_mapping_contract_rejects_dynamic_fields_absent_from_actual_spec(
    monkeypatch,
    tmp_path,
    target_field: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with pytest.MonkeyPatch.context():
        with CliRunner().isolated_filesystem(temp_dir=tmp_path):
            spec = StrategySpec.template(strategy_id="equal_weight", hypothesis="actual fields only")
            result = validate_mapping_contract(_mapped_strategy_target(target_field), spec=spec)

    assert result["status"] == "fail"
    assert any(error["path"] == "field_mappings[0].target_field" for error in result["errors"])


@pytest.mark.parametrize(
    "target_field",
    [
        "portfolio",
        "portfolio.params",
        "execution",
        "universe.symbols[0]",
    ],
)
def test_mapping_contract_builder_pass_rejects_non_leaf_targets_from_actual_spec(
    target_field: str,
) -> None:
    spec = StrategySpec.template(strategy_id="leaf_targets", hypothesis="map exact effective fields")

    result = validate_mapping_contract_for_builder_pass(
        _mapped_strategy_target(target_field),
        spec=spec,
    )

    assert result["status"] == "fail"
    assert any(error["path"] == "field_mappings[0].target_field" for error in result["errors"])


@pytest.mark.parametrize(
    "target_field",
    [
        "portfolio.type",
        "portfolio.params.n",
        "portfolio.params.score_col",
    ],
)
def test_mapping_contract_accepts_fields_present_in_actual_top_n_spec(
    monkeypatch,
    tmp_path,
    target_field: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with CliRunner().isolated_filesystem(temp_dir=tmp_path):
        spec = StrategySpec.template(strategy_id="top_n", hypothesis="rank candidates")
        spec.portfolio.type = "TopNRanking"
        spec.portfolio.params = {"n": 5, "score_col": "momentum"}
        result = validate_mapping_contract(_mapped_strategy_target(target_field), spec=spec)

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_mapping_contract_builder_pass_rejects_mapped_confirmation_flag(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    payload = _mapped_strategy_target("portfolio.type")
    payload["field_mappings"][0]["confirmation_required"] = True
    with CliRunner().isolated_filesystem(temp_dir=tmp_path):
        base_result = validate_mapping_contract(payload)
        pass_result = validate_mapping_contract_for_builder_pass(payload)

    assert base_result["status"] == "fail"
    assert pass_result["status"] == "fail"
    assert any(error["path"] == "field_mappings[0].confirmation_required" for error in pass_result["errors"])


def test_mapping_contract_builder_pass_accepts_fully_mapped_false_confirmation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    payload = _mapped_strategy_target("portfolio.type")
    spec = StrategySpec.template(strategy_id="bound_builder_pass", hypothesis="builder pass uses exact fields")
    with CliRunner().isolated_filesystem(temp_dir=tmp_path):
        result = validate_mapping_contract_for_builder_pass(payload, spec=spec)

    assert result["status"] == "pass"
    assert result["errors"] == []


@pytest.mark.parametrize(
    "target_field",
    [
        "portfolio.type",
        "signal.indicators.roc.params",
        "signal.indicators.roc.params.period",
    ],
)
def test_mapping_contract_builder_pass_rejects_unbound_effective_targets(target_field: str) -> None:
    result = validate_mapping_contract_for_builder_pass(_mapped_strategy_target(target_field))

    assert result["status"] == "fail"
    assert any(
        error["path"] == "$" and "concrete effective leaf inventory" in error["message"]
        for error in result["errors"]
    )


def test_mapping_contract_builder_pass_accepts_exact_dynamic_leaf_from_bound_spec() -> None:
    spec = StrategySpec.template(strategy_id="bound_roc_leaf", hypothesis="map an exact configured ROC leaf")
    spec.signal.indicators = {
        "roc": IndicatorDef(type="ROC", params={"column": "close", "period": 20})
    }

    result = validate_mapping_contract_for_builder_pass(
        _mapped_strategy_target("signal.indicators.roc.params.period"),
        spec=spec,
    )

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_mapping_contract_builder_pass_rejects_dynamic_container_from_bound_spec() -> None:
    spec = StrategySpec.template(strategy_id="bound_roc_container", hypothesis="map exact configured leaves only")
    spec.signal.indicators = {
        "roc": IndicatorDef(type="ROC", params={"column": "close", "period": 20})
    }

    result = validate_mapping_contract_for_builder_pass(
        _mapped_strategy_target("signal.indicators.roc.params"),
        spec=spec,
    )

    assert result["status"] == "fail"
    assert any(error["path"] == "field_mappings[0].target_field" for error in result["errors"])


@pytest.mark.parametrize("source_fields", [None, []])
def test_mapping_contract_builder_pass_rejects_missing_or_empty_source_inventory(source_fields) -> None:
    payload = _mapped_strategy_target("portfolio.type")
    if source_fields is None:
        payload.pop("source_fields")
    else:
        payload["source_fields"] = source_fields

    result = validate_mapping_contract_for_builder_pass(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "source_fields" for error in result["errors"])


def test_mapping_contract_builder_pass_rejects_incomplete_source_inventory_coverage() -> None:
    payload = _mapped_strategy_target("portfolio.type")
    payload["source_fields"] = ["source.dynamic_field", "source.unmapped_field"]

    result = validate_mapping_contract_for_builder_pass(payload)

    assert result["status"] == "fail"
    assert any(
        error["path"] == "source_fields[1]" and "missing field_mappings row" in error["message"]
        for error in result["errors"]
    )


def test_mapping_contract_builder_pass_rejects_duplicate_source_inventory_field() -> None:
    payload = _mapped_strategy_target("portfolio.type")
    payload["source_fields"] = ["source.dynamic_field", "source.dynamic_field"]

    result = validate_mapping_contract_for_builder_pass(payload)

    assert result["status"] == "fail"
    assert any(error["path"] == "source_fields[1]" and "duplicate" in error["message"] for error in result["errors"])


def test_mapping_contract_builder_pass_rejects_invented_source_mapping_row() -> None:
    payload = _mapped_strategy_target("portfolio.type")
    payload["source_fields"] = ["source.actual_field"]

    result = validate_mapping_contract_for_builder_pass(payload)

    assert result["status"] == "fail"
    assert any(
        error["path"] == "field_mappings[0].source_field" and "not declared in source_fields" in error["message"]
        for error in result["errors"]
    )


def test_mapping_contract_builder_pass_accepts_exact_source_inventory_coverage() -> None:
    payload = _mapped_strategy_target("portfolio.type")
    payload["source_fields"] = ["source.dynamic_field"]

    result = validate_mapping_contract_for_builder_pass(
        payload,
        effective_field_paths={"portfolio.type"},
    )

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_mapping_contract_builder_pass_accepts_exact_canonical_idea_inventory() -> None:
    payload = _mapped_strategy_target("portfolio.type")
    idea_brief = {
        "schema_version": 1,
        "conversation_hash": "sha256:" + "a" * 16,
        "source": {"dynamic_field": 5},
    }

    result = validate_mapping_contract_for_builder_pass(
        payload,
        effective_field_paths={"portfolio.type"},
        idea_brief=idea_brief,
    )

    assert result["status"] == "pass"
    assert result["errors"] == []


def test_mapping_contract_builder_pass_requires_canonical_idea_inventory() -> None:
    payload = _mapped_strategy_target("portfolio.type")
    payload["source_format"] = "strategy_idea_brief"

    result = validate_mapping_contract_for_builder_pass(
        payload,
        effective_field_paths={"portfolio.type"},
    )

    assert result["status"] == "fail"
    assert any(error["path"] == "idea_brief" for error in result["errors"])


def test_mapping_contract_builder_pass_rejects_source_inventory_addition() -> None:
    payload = _mapped_strategy_target("portfolio.type")
    payload["source_fields"] = ["source.dynamic_field", "source.invented"]
    payload["field_mappings"].append(
        {
            "source_field": "source.invented",
            "target_field": "",
            "semantic": "metadata",
            "status": "excluded_non_material",
            "confirmation_required": False,
            "blocking": False,
            "reason": "This field is not present in the canonical brief.",
        }
    )

    result = validate_mapping_contract_for_builder_pass(
        payload,
        effective_field_paths={"portfolio.type"},
        idea_brief={"source": {"dynamic_field": 5}},
    )

    assert result["status"] == "fail"
    assert any(
        error["path"] == "source_fields[1]" and "canonical strategy idea brief" in error["message"]
        for error in result["errors"]
    )


def test_mapping_contract_builder_pass_rejects_source_inventory_removal() -> None:
    payload = _mapped_strategy_target("portfolio.type")

    result = validate_mapping_contract_for_builder_pass(
        payload,
        effective_field_paths={"portfolio.type"},
        idea_brief={
            "source": {
                "dynamic_field": 5,
                "required_field": {"window": 20},
            }
        },
    )

    assert result["status"] == "fail"
    assert any(
        error["path"] == "source_fields" and "source.required_field.window" in error["message"]
        for error in result["errors"]
    )


@pytest.mark.parametrize(
    ("status", "blocking", "target_field"),
    [
        ("mapped", False, "portfolio.type"),
        ("excluded_non_material", False, ""),
        ("unsupported", False, ""),
        ("unsupported", True, ""),
        ("blocked", True, ""),
    ],
)
def test_mapping_contract_builder_pass_rejects_unsupported_semantic_combinations(
    status: str,
    blocking: bool,
    target_field: str,
) -> None:
    payload = _mapped_strategy_target("portfolio.type")
    payload["field_mappings"][0].update(
        {
            "semantic": "unsupported",
            "status": status,
            "blocking": blocking,
            "target_field": target_field,
        }
    )

    result = validate_mapping_contract_for_builder_pass(
        payload,
        effective_field_paths={"portfolio.type"},
    )

    assert result["status"] == "fail"
    assert any(
        error["path"] in {"field_mappings[0].semantic", "field_mappings[0].status", "field_mappings[0].blocking"}
        for error in result["errors"]
    )


def test_mapping_contract_rejects_duplicate_source_field() -> None:
    payload = {
        "schema_version": 1,
        "source_format": "ebacktestcraft_yaml",
        "field_mappings": [
            {
                "source_field": "market.cross_calendar_policy",
                "target_field": "market.calendar",
                "semantic": "strategy",
                "status": "mapped",
                "confirmation_required": False,
                "blocking": False,
                "reason": "Primary calendar is mapped.",
            },
            {
                "source_field": "market.cross_calendar_policy",
                "target_field": "data.provider",
                "semantic": "strategy",
                "status": "mapped",
                "confirmation_required": False,
                "blocking": False,
                "reason": "Duplicate row should be folded into a single mapping row.",
            },
        ],
    }

    result = validate_mapping_contract(payload)

    assert result["status"] == "fail"
    assert any(
        error["path"] == "field_mappings[1].source_field" and "duplicate source_field" in error["message"]
        for error in result["errors"]
    )
