"""Tests for deterministic component catalog export."""

from __future__ import annotations

import json
from decimal import Decimal

from oxq.core.component_catalog import _params_for, build_component_catalog, component_catalog_json


def test_component_catalog_contains_registered_components_and_recipes() -> None:
    catalog = build_component_catalog()

    assert catalog["schema_version"] == 1
    assert catalog["catalog_hash"].startswith("sha256:")
    assert catalog["recipe_catalog_hash"].startswith("sha256:")

    indicators = {item["name"]: item for item in catalog["indicators"]}
    signals = {item["name"]: item for item in catalog["signals"]}
    portfolios = {item["name"]: item for item in catalog["portfolios"]}
    recipes = {item["name"]: item for item in catalog["recipes"]}

    assert "NdayReturn" in indicators
    assert "20日收益率" in indicators["NdayReturn"]["aliases"]
    assert "dependencies" in indicators["NdayReturn"]
    assert "RollingVolatility" in indicators
    assert "Ratio" in indicators
    assert indicators["Ratio"]["params"]["col_a"]["required"] is True
    assert indicators["Ratio"]["params"]["col_a"]["semantic_required"] is True
    assert indicators["Ratio"]["params"]["col_b"]["required"] is True
    assert signals["Composite"]["params"]["signals"]["required"] is True
    assert signals["Composite"]["params"]["signals"]["semantic_required"] is True
    assert "TopNRanking" in portfolios
    assert "roc_timing" in recipes
    assert "sma_golden_cross" in recipes
    assert "top_n_positive_momentum_rotation" in recipes
    assert "top_n_normalized_weights" in recipes
    assert recipes["volatility_adjusted_momentum"]["canonical_spec"]["signal"]["indicators"]["vol_adj_momentum"]["type"] == "Ratio"


def test_component_catalog_json_is_stable() -> None:
    first = component_catalog_json()
    second = component_catalog_json()

    assert first == second
    parsed = json.loads(first)
    assert parsed["catalog_hash"] == build_component_catalog()["catalog_hash"]


def test_component_catalog_params_are_json_safe() -> None:
    class RichDefaultIndicator:
        name = "RichDefaultIndicator"

        def compute(
            self,
            mktdata,
            threshold=Decimal("1.25"),
            choices={"b", "a"},
        ):
            return mktdata["close"]

    params = _params_for("indicators", RichDefaultIndicator)

    assert params["threshold"]["default"] == "1.25"
    assert params["choices"]["default"] == ["a", "b"]
    json.dumps(params)
