from oxq.spec.mapping_contract import validate_mapping_contract, validate_mapping_contract_for_builder_pass


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
        and "unsupported strategy semantics require blocking=true" in error["message"]
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
        and "builder pass requires strategy mappings to be mapped and non-blocking" in error["message"]
        for error in pass_result["errors"]
    )


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
