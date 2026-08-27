"""Immutable example provider source used by the digest conformance fixture."""

from collections.abc import Sequence


def sma(values: Sequence[float], window: int = 20) -> list[float | None]:
    """Return a simple moving average with explicit warmup values."""

    if window < 1:
        raise ValueError("window must be positive")
    result: list[float | None] = []
    for index in range(len(values)):
        if index + 1 < window:
            result.append(None)
            continue
        sample = values[index + 1 - window : index + 1]
        result.append(sum(sample) / window)
    return result
