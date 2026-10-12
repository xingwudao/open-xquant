"""Weighted sum of existing mktdata columns."""

from __future__ import annotations

import math

import pandas as pd


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class LinearCombination:
    """``sum_i(weights[i] * mktdata[columns[i]]) + constant``.

    Columns may reference earlier-declared indicators: the engine computes
    indicators in declaration order and writes each output back to mktdata.
    """

    name = "LinearCombination"
    formula = r"Y_t = c + \sum_i w_i X_{i,t}"

    def compute(
        self,
        mktdata: pd.DataFrame,
        columns: list[str] | None = None,
        weights: list[float] | None = None,
        constant: float = 0.0,
    ) -> pd.Series:
        """Return the weighted sum of ``columns`` plus ``constant``."""
        if not isinstance(columns, list) or not columns or not all(isinstance(c, str) and c for c in columns):
            raise ValueError("LinearCombination requires a non-empty list of column names")
        if len(set(columns)) != len(columns):
            raise ValueError("LinearCombination columns must be unique")
        if not isinstance(weights, list) or len(weights) != len(columns):
            raise ValueError("LinearCombination weights must match columns one-to-one")
        if not all(_is_finite_number(w) for w in [*weights, constant]):
            raise ValueError("LinearCombination weights and constant must be finite numbers")
        result = pd.Series(float(constant), index=mktdata.index)
        for column, weight in zip(columns, weights, strict=True):
            result = result + float(weight) * mktdata[column].astype(float)
        return result
