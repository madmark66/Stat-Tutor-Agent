"""Deterministic statistical calculation engine (Left Brain)."""

import math
from typing import Any, Sequence


def _validate_numeric_list(numbers: Sequence[Any]) -> list[float]:
    """Validate that input is a list/tuple of at least 2 finite numbers (excluding booleans)."""
    if not isinstance(numbers, (list, tuple)):
        raise ValueError("輸入必須是數字列表")

    if len(numbers) < 2:
        raise ValueError("樣本變異數至少需要兩個數字")

    values: list[float] = []
    for item in numbers:
        # In Python, isinstance(True, int) is True, so explicitly reject booleans.
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("列表中的所有元素都必須是有效數字")
        val = float(item)
        if not math.isfinite(val):
            raise ValueError("數據不能包含 NaN 或無限大")
        values.append(val)

    return values


def calculate_sample_variance(numbers: Sequence[float | int]) -> dict[str, Any]:
    """
    Calculate deterministic variance metrics with clear separation between ddof=0 and ddof=1.

    Returns a dictionary containing:
    - status: "success"
    - sample_size: n
    - degrees_of_freedom: n - 1
    - mean: sample arithmetic mean
    - sum_squared_deviations: sum((x - mean)^2)
    - variance_ddof1: unbiased sample variance (divided by n - 1)
    - variance_ddof0: population / biased sample variance (divided by n)
    - sample_variance: backward-compatible alias equal to variance_ddof1
    """
    values = _validate_numeric_list(numbers)
    n = len(values)
    df = n - 1
    mean_val = float(sum(values) / n)
    ss_val = float(sum((x - mean_val) ** 2 for x in values))

    variance_ddof1 = float(ss_val / df)
    variance_ddof0 = float(ss_val / n)

    return {
        "status": "success",
        "sample_size": n,
        "degrees_of_freedom": df,
        "mean": mean_val,
        "sum_squared_deviations": ss_val,
        "variance_ddof1": variance_ddof1,
        "variance_ddof0": variance_ddof0,
        "sample_variance": variance_ddof1,
    }
