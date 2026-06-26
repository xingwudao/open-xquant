"""Deterministic component catalog export for Agent strategy construction."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from typing import Any

from oxq.core.registry import (
    list_indicator_metadata,
    list_indicators,
    list_portfolio_optimizers,
    list_rules,
    list_signals,
)

CATALOG_SCHEMA_VERSION = 1


_COMPONENT_ALIASES: dict[str, dict[str, list[str]]] = {
    "indicators": {
        "NdayReturn": ["n-day return", "20日收益率", "动量收益率", "N日收益率", "收益率动量"],
        "RollingVolatility": ["rolling volatility", "20日波动率", "N日波动率", "滚动波动率"],
        "Ratio": ["ratio", "比值", "相除", "除以"],
        "ROC": ["rate of change", "变动率", "价格变化率"],
        "SMA": ["simple moving average", "moving average", "均线", "简单移动平均"],
    },
    "signals": {
        "Crossover": ["cross", "crossover", "金叉", "上穿"],
        "ROCTiming": ["roc timing", "roc reversal", "动量择时", "反转择时"],
        "Threshold": ["threshold", "阈值", "大于小于", "条件过滤"],
    },
    "portfolios": {
        "TopNRanking": ["top n", "topn", "取TopN", "排名前N", "归一化权重", "按得分归一化"],
        "EqualWeight": ["equal weight", "等权", "等权重"],
        "SignalToPosition": ["signal to position", "BUY SELL HOLD", "买卖持仓映射"],
    },
    "rules": {
        "StopLossRule": ["stop loss", "止损"],
        "TakeProfitRule": ["take profit", "止盈"],
        "RebalanceFrequencyRule": ["rebalance frequency", "调仓频率"],
        "MaxHoldingsRule": ["max holdings", "最大持仓数"],
    },
}


_COMPONENT_FORMULAS: dict[str, dict[str, str]] = {
    "portfolios": {
        "EqualWeight": "Assign equal weight to each symbol passing boolean signal filters.",
        "TopNRanking": "Rank symbols by score_col, select top n, and normalize selected scores into target weights.",
        "SignalToPosition": "Map categorical BUY/SELL/HOLD signal values to target weights with HOLD maintaining state.",
    },
    "signals": {
        "Crossover": "fast.shift(1) <= slow.shift(1) and fast > slow",
        "Threshold": "column relationship threshold",
        "ROCTiming": "BUY below bottom threshold, SELL above top threshold, HOLD otherwise",
    },
    "rules": {
        "StopLossRule": "Exit when unrealized loss exceeds threshold.",
        "TakeProfitRule": "Exit when unrealized profit exceeds threshold.",
        "RebalanceFrequencyRule": "Permit rebalancing only every interval_days trading sessions.",
        "MaxHoldingsRule": "Limit the number of simultaneously held positions.",
    },
}


_COMPONENT_CATEGORIES: dict[str, dict[str, str]] = {
    "portfolios": {
        "EqualWeight": "allocation",
        "TopNRanking": "ranking",
        "SignalToPosition": "execution_mapping",
        "RiskParity": "risk_allocation",
        "Kelly": "position_sizing",
        "PctEquity": "position_sizing",
    },
    "signals": {
        "Crossover": "technical",
        "Threshold": "filter",
        "ROCTiming": "timing",
        "Comparison": "filter",
        "Composite": "composition",
        "Formula": "custom_expression",
        "Peak": "pattern",
        "Timestamp": "calendar",
    },
    "rules": {
        "BlacklistRule": "constraint",
        "DailyLossLimitRisk": "risk",
        "ExitRule": "exit",
        "MaxDrawdownRisk": "risk",
        "MaxHoldingsRule": "constraint",
        "RebalanceFrequencyRule": "constraint",
        "StopLossRule": "exit",
        "TakeProfitRule": "exit",
        "TrailingStopRule": "exit",
    },
}


CANONICAL_RECIPES: list[dict[str, Any]] = [
    {
        "name": "roc_timing",
        "aliases": ["ROC timing", "roc择时", "roc timing signal", "ROC reversal timing", "动量反转择时"],
        "definition": "ROC(period=n) feeds ROCTiming BUY/SELL/HOLD, then SignalToPosition maps signal to positions.",
        "required_components": {
            "indicators": ["ROC"],
            "signals": ["ROCTiming"],
            "portfolios": ["SignalToPosition"],
            "rules": [],
        },
        "canonical_spec": {
            "signal": {
                "indicators": {
                    "roc_n": {
                        "type": "ROC",
                        "params": {"column": "close", "period": "$period"},
                    }
                },
                "rules": {
                    "timing": {
                        "type": "ROCTiming",
                        "output_domain": ["BUY", "SELL", "HOLD"],
                        "params": {
                            "column": "roc_n",
                            "mode": "$mode",
                            "bottom": "$bottom",
                            "top": "$top",
                        },
                    }
                },
            },
            "portfolio": {
                "type": "SignalToPosition",
                "params": {"signal": "timing"},
            },
        },
    },
    {
        "name": "sma_golden_cross",
        "aliases": ["SMA 金叉", "均线金叉", "SMA crossover", "moving average crossover", "golden cross"],
        "definition": "SMA(fast_period) crosses above SMA(slow_period).",
        "required_components": {
            "indicators": ["SMA"],
            "signals": ["Crossover"],
            "portfolios": ["EqualWeight"],
            "rules": ["ExitRule"],
        },
        "canonical_spec": {
            "signal": {
                "indicators": {
                    "sma_fast": {
                        "type": "SMA",
                        "params": {"column": "close", "period": "$fast_period"},
                    },
                    "sma_slow": {
                        "type": "SMA",
                        "params": {"column": "close", "period": "$slow_period"},
                    },
                },
                "rules": {
                    "golden_cross": {
                        "type": "Crossover",
                        "params": {"fast": "sma_fast", "slow": "sma_slow"},
                    }
                },
            },
            "portfolio": {
                "type": "EqualWeight",
                "params": {},
            },
        },
    },
    {
        "name": "top_n_positive_momentum_rotation",
        "aliases": ["TopN 正动量轮动", "top n positive momentum rotation", "正收益动量轮动", "TopN momentum rotation"],
        "definition": "NdayReturn(period=n) score, select positive top n, normalize score weights.",
        "required_components": {
            "indicators": ["NdayReturn"],
            "signals": [],
            "portfolios": ["TopNRanking"],
            "rules": [],
        },
        "canonical_spec": {
            "signal": {
                "indicators": {
                    "momentum_n": {
                        "type": "NdayReturn",
                        "params": {"column": "close", "period": "$period"},
                    }
                }
            },
            "portfolio": {
                "type": "TopNRanking",
                "params": {
                    "score_col": "momentum_n",
                    "n": "$n",
                    "filter_negative": True,
                    "max_weight": 1.0,
                },
            },
        },
    },
    {
        "name": "volatility_adjusted_momentum",
        "aliases": ["波动率调整动量", "risk adjusted momentum", "volatility adjusted momentum", "收益率除以波动率"],
        "definition": "NdayReturn(period=n) / RollingVolatility(period=n)",
        "required_components": {
            "indicators": ["NdayReturn", "RollingVolatility", "Ratio"],
            "signals": [],
            "portfolios": [],
            "rules": [],
        },
        "canonical_spec": {
            "signal": {
                "indicators": {
                    "ret_n": {
                        "type": "NdayReturn",
                        "params": {"column": "close", "period": "$period"},
                    },
                    "vol_n": {
                        "type": "RollingVolatility",
                        "params": {"column": "close", "period": "$period"},
                    },
                    "vol_adj_momentum": {
                        "type": "Ratio",
                        "params": {"col_a": "ret_n", "col_b": "vol_n"},
                    },
                }
            }
        },
    },
    {
        "name": "top_n_normalized_weights",
        "aliases": ["取TopN，归一化权重", "top n normalized weights", "rank top n and normalize", "TopN ranking allocation"],
        "definition": "Rank by score_col, select top n, normalize score weights.",
        "required_components": {
            "indicators": [],
            "signals": [],
            "portfolios": ["TopNRanking"],
            "rules": [],
        },
        "canonical_spec": {
            "portfolio": {
                "type": "TopNRanking",
                "params": {
                    "score_col": "$score_col",
                    "n": "$n",
                    "filter_negative": True,
                    "max_weight": 1.0,
                },
            }
        },
    },
]


def build_component_catalog(component_manifests: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return a deterministic catalog of registered components and recipes."""
    catalog: dict[str, Any] = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "indicators": _component_entries("indicators", list_indicators(), list_indicator_metadata()),
        "signals": _component_entries("signals", list_signals()),
        "portfolios": _component_entries("portfolios", list_portfolio_optimizers()),
        "rules": _component_entries("rules", list_rules()),
        "recipes": sorted(CANONICAL_RECIPES, key=lambda item: item["name"]),
    }
    for manifest in component_manifests or []:
        _apply_manifest_metadata(catalog, manifest)
    catalog["recipe_catalog_hash"] = _stable_hash(catalog["recipes"])
    catalog["catalog_hash"] = _catalog_hash(catalog)
    return catalog


def component_catalog_json(catalog: Mapping[str, Any] | None = None) -> str:
    """Serialize a component catalog with stable key and list ordering."""
    return json.dumps(catalog or build_component_catalog(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _component_entries(
    kind: str,
    registry: Mapping[str, type],
    metadata: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    return [_component_entry(kind, name, cls, metadata or {}) for name, cls in sorted(registry.items())]


def _component_entry(
    kind: str,
    name: str,
    cls: type,
    metadata: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    info = dict(metadata.get(name, {}))
    formula = getattr(cls, "formula", "") or _COMPONENT_FORMULAS.get(kind, {}).get(name, "")
    category = info.get("category") or _COMPONENT_CATEGORIES.get(kind, {}).get(name, "")
    description = info.get("description") or inspect.getdoc(cls) or ""
    return {
        "name": name,
        "aliases": sorted(_COMPONENT_ALIASES.get(kind, {}).get(name, []), key=str.casefold),
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "category": category,
        "description": description,
        "dependencies": _dependencies_for(cls),
        "formula": formula,
        "output_type": _output_type(kind, name),
        "params": _params_for(kind, cls),
        "source": "builtin" if cls.__module__.startswith("oxq.") else "custom",
        "source_type": info.get("source_type", ""),
    }


def _params_for(kind: str, cls: type) -> dict[str, Any]:
    method_name = "compute" if kind in {"indicators", "signals"} else "__init__"
    try:
        signature = inspect.signature(getattr(cls, method_name))
    except (TypeError, ValueError, AttributeError):
        return {}
    params: dict[str, Any] = {}
    skip = {"self", "mktdata"}
    for name, param in signature.parameters.items():
        if name in skip or param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            continue
        sentinel_required = param.default == "" or (
            kind == "signals" and getattr(cls, "name", cls.__name__) == "Composite" and name == "signals"
        )
        required = param.default is inspect.Parameter.empty or sentinel_required
        entry: dict[str, Any] = {
            "required": required,
        }
        if sentinel_required:
            entry["semantic_required"] = True
            entry["required_reason"] = "sentinel default is not a usable spec value"
        if param.default is not inspect.Parameter.empty:
            entry["default"] = _json_safe(param.default)
        if param.annotation is not inspect.Parameter.empty:
            entry["type"] = _annotation_name(param.annotation)
        params[name] = entry
    return params


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        values = [_json_safe(item) for item in value]
        return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return str(value)


def _annotation_name(annotation: object) -> str:
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def _dependencies_for(cls: type) -> dict[str, list[str]]:
    required_indicators = getattr(cls, "required_indicators", {})
    if isinstance(required_indicators, Mapping):
        indicator_names = sorted(str(name) for name in required_indicators)
    else:
        indicator_names = sorted(str(name) for name in required_indicators or [])
    return {
        "columns": sorted(str(name) for name in getattr(cls, "depends_on", ()) or []),
        "required_indicators": indicator_names,
    }


def _output_type(kind: str, name: str) -> str:
    if kind == "indicators":
        return "numeric_series"
    if kind == "signals":
        return "categorical_series" if name == "ROCTiming" else "boolean_series"
    if kind == "portfolios":
        return "target_weights"
    return "rule_result"


def _apply_manifest_metadata(catalog: dict[str, Any], manifest: Mapping[str, Any]) -> None:
    manifest_path = str(manifest.get("_manifest_path") or manifest.get("manifest_path") or "component_manifest.json")
    bundle_hash = str(manifest.get("bundle_hash") or "")
    for raw_component in manifest.get("components", []):
        if not isinstance(raw_component, Mapping):
            continue
        kind = str(raw_component.get("kind") or "")
        section = {
            "Indicator": "indicators",
            "Signal": "signals",
            "PortfolioOptimizer": "portfolios",
            "Rule": "rules",
        }.get(kind)
        name = raw_component.get("name")
        if section is None or not isinstance(name, str) or not name:
            continue
        entries = catalog.get(section)
        if not isinstance(entries, list):
            continue
        entry = next((item for item in entries if isinstance(item, dict) and item.get("name") == name), None)
        if entry is None:
            entry = {"name": name}
            entries.append(entry)
        entry.update(
            {
                "class": ".".join(
                    part
                    for part in [
                        str(raw_component.get("module") or ""),
                        str(raw_component.get("class") or ""),
                    ]
                    if part
                ),
                "module": raw_component.get("module", ""),
                "source": "workspace_extension",
                "source_type": "workspace_extension",
                "manifest_path": manifest_path,
                "bundle_hash": bundle_hash,
                "output_domain": raw_component.get("output_domain", []),
            }
        )
        if isinstance(raw_component.get("parameters"), Mapping):
            _merge_manifest_parameters(entry, raw_component["parameters"])
        if "description" in raw_component:
            entry["description"] = raw_component["description"]
        if "formula" in raw_component:
            entry["formula"] = raw_component["formula"]
    for section in ("indicators", "signals", "portfolios", "rules"):
        if isinstance(catalog.get(section), list):
            catalog[section] = sorted(catalog[section], key=lambda item: str(item.get("name", "")))


def _merge_manifest_parameters(entry: dict[str, Any], parameters: Mapping[str, Any]) -> None:
    manifest_parameters = dict(parameters)
    entry["manifest_parameters"] = manifest_parameters
    existing = entry.get("params")
    if not isinstance(existing, Mapping):
        entry["params"] = {
            name: {"default": value, "source": "manifest"}
            for name, value in manifest_parameters.items()
        }
        return

    merged: dict[str, Any] = dict(existing)
    for name, value in manifest_parameters.items():
        current = merged.get(name)
        if isinstance(current, Mapping):
            annotated = dict(current)
            annotated["manifest_value"] = value
            merged[name] = annotated
        else:
            merged[name] = {"default": value, "source": "manifest"}
    entry["params"] = merged


def _catalog_hash(catalog: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in catalog.items() if key not in {"catalog_hash", "recipe_catalog_hash"}}
    return _stable_hash(payload)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
