"""Small numeric helpers shared by the analysis modules."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


def robust_normalize(
    values: ArrayLike,
    *,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    min_range: float = 1e-6,
) -> NDArray[np.float32]:
    """Scale values to ``[0, 1]`` between two percentiles of their own distribution.

    Outliers saturate instead of squashing everything else. When the distribution
    is flat (range below ``min_range``) there is no signal, so everything maps to 0.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros(0, dtype=np.float32)
    low = float(np.percentile(array, low_percentile))
    high = float(np.percentile(array, high_percentile))
    if high - low < min_range:
        return np.zeros(array.shape, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def zscores(values: ArrayLike, *, axis: int = 0, min_std: float = 1e-6) -> NDArray[np.float32]:
    """Standard scores along ``axis``; constant slices yield zeros."""
    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=axis, keepdims=True)
    std = array.std(axis=axis, keepdims=True)
    safe = np.where(std < min_std, 1.0, std)
    return np.where(std < min_std, 0.0, (array - mean) / safe).astype(np.float32)
