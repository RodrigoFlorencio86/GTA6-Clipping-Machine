"""Editorial highlight score.

    score = 0.35 * semantic_similarity
          + 0.20 * audio_intensity
          + 0.15 * visual_motion
          + 0.15 * reaction_presence
          + 0.10 * novelty
          + 0.05 * duration_fit

Every signal is in ``[0, 1]``; the weights come from configuration and sum to 1.
The score ranks candidates for a human reviewer. It predicts nothing about reach.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from clipping_machine.analysis.normalization import clamp01
from clipping_machine.config import ScoreWeights

SIGNAL_NAMES: tuple[str, ...] = (
    "semantic_similarity",
    "audio_intensity",
    "visual_motion",
    "reaction_presence",
    "novelty",
    "duration_fit",
)


@dataclass(frozen=True)
class ScoreSignals:
    semantic_similarity: float
    audio_intensity: float
    visual_motion: float
    reaction_presence: float
    novelty: float
    duration_fit: float

    def clamped(self) -> ScoreSignals:
        return ScoreSignals(**{name: clamp01(value) for name, value in asdict(self).items()})


@dataclass(frozen=True)
class ScoreBreakdown:
    total: float
    signals: ScoreSignals
    contributions: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        """Flat mapping persisted with each clip."""
        return {name: round(value, 4) for name, value in asdict(self.signals).items()}


def duration_fit(duration_ms: int, target_ms: tuple[int, int], min_ms: int, max_ms: int) -> float:
    """1.0 inside the target range, falling off linearly outside it.

    Below the target it falls to 0.5 at the minimum clip length; above it, to 0 at
    the maximum clip length.
    """
    low, high = target_ms
    if duration_ms <= 0:
        return 0.0
    if low <= duration_ms <= high:
        return 1.0
    if duration_ms < low:
        if duration_ms < min_ms:
            return clamp01(0.5 * duration_ms / max(min_ms, 1))
        span = max(low - min_ms, 1)
        return clamp01(0.5 + 0.5 * (duration_ms - min_ms) / span)
    span = max(max_ms - high, 1)
    return clamp01(1.0 - (duration_ms - high) / span)


def temporal_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    """Fraction of the shorter interval covered by the intersection."""
    intersection = min(end_a, end_b) - max(start_a, start_b)
    shortest = min(end_a - start_a, end_b - start_b)
    if intersection <= 0 or shortest <= 0:
        return 0.0
    return clamp01(intersection / shortest)


def novelty(
    start_ms: int,
    end_ms: int,
    vector: NDArray[np.float32] | None,
    previous_intervals: Sequence[tuple[int, int]],
    previous_vectors: Sequence[NDArray[np.float32]],
    *,
    similarity_floor: float = 0.6,
) -> float:
    """How different a candidate is from clips already made (1 = brand new).

    Redundancy is the larger of (a) temporal overlap with an earlier clip of the same
    video and (b) embedding similarity to any earlier clip, rescaled so that only
    cosine values above ``similarity_floor`` count as repetition.
    """
    redundancy = 0.0
    for other_start, other_end in previous_intervals:
        redundancy = max(redundancy, temporal_overlap(start_ms, end_ms, other_start, other_end))
    if vector is not None:
        norm = float(np.linalg.norm(vector))
        for other in previous_vectors:
            other_norm = float(np.linalg.norm(other))
            if norm == 0.0 or other_norm == 0.0:
                continue
            cosine = float(np.dot(vector, other) / (norm * other_norm))
            redundancy = max(
                redundancy, clamp01((cosine - similarity_floor) / (1.0 - similarity_floor))
            )
    return clamp01(1.0 - redundancy)


def compute_score(signals: ScoreSignals, weights: ScoreWeights) -> ScoreBreakdown:
    clamped = signals.clamped()
    weight_map = weights.model_dump()
    values = asdict(clamped)
    contributions = {name: weight_map[name] * values[name] for name in SIGNAL_NAMES}
    return ScoreBreakdown(
        total=clamp01(sum(contributions.values())), signals=clamped, contributions=contributions
    )
