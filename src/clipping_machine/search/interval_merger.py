"""Turns scattered segment hits into clip-sized intervals.

Rules (docs/ARCHITECTURE.md, "Interval merging"):

1. Hits closer than ``merge_gap`` are merged into one interval.
2. Intervals shorter than the minimum clip length are expanded around their centre.
3. Intervals longer than the maximum clip length are split into equal parts no
   longer than the upper end of the target range.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    """One segment returned by the vector search."""

    segment_id: str
    start_ms: int
    end_ms: int
    similarity: float


@dataclass(frozen=True)
class MergedInterval:
    start_ms: int
    end_ms: int
    hits: tuple[Hit, ...]

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def best_hit(self) -> Hit:
        return max(self.hits, key=lambda hit: hit.similarity)

    @property
    def similarity(self) -> float:
        """Peak-weighted similarity: one strong hit matters more than a weak tail."""
        values = [hit.similarity for hit in self.hits]
        return 0.7 * max(values) + 0.3 * (sum(values) / len(values))


def merge_hits(hits: Sequence[Hit], merge_gap_ms: int) -> list[MergedInterval]:
    """Merge hits whose gap is strictly smaller than ``merge_gap_ms``.

    Overlapping hits (negative gap) always merge.
    """
    merged: list[MergedInterval] = []
    for hit in sorted(hits, key=lambda item: (item.start_ms, item.end_ms)):
        if merged and hit.start_ms - merged[-1].end_ms < merge_gap_ms:
            last = merged[-1]
            merged[-1] = MergedInterval(
                last.start_ms, max(last.end_ms, hit.end_ms), (*last.hits, hit)
            )
        else:
            merged.append(MergedInterval(hit.start_ms, hit.end_ms, (hit,)))
    return merged


def expand_to_minimum(
    interval: MergedInterval, min_ms: int, video_duration_ms: int
) -> MergedInterval:
    """Grow a short interval symmetrically to ``min_ms``, staying inside the video."""
    if interval.duration_ms >= min_ms:
        return interval
    target = min(min_ms, video_duration_ms)
    missing = target - interval.duration_ms
    start = interval.start_ms - missing // 2
    end = start + target
    if start < 0:
        start, end = 0, target
    if end > video_duration_ms:
        end = video_duration_ms
        start = max(0, end - target)
    return MergedInterval(start, end, interval.hits)


def split_long(interval: MergedInterval, max_ms: int, target_max_ms: int) -> list[MergedInterval]:
    """Split an interval longer than ``max_ms`` into equal parts <= ``target_max_ms``."""
    if interval.duration_ms <= max_ms:
        return [interval]
    parts = math.ceil(interval.duration_ms / max(target_max_ms, 1))
    length = interval.duration_ms / parts
    pieces: list[MergedInterval] = []
    for index in range(parts):
        start = interval.start_ms + round(index * length)
        end = (
            interval.end_ms
            if index == parts - 1
            else interval.start_ms + round((index + 1) * length)
        )
        members = tuple(hit for hit in interval.hits if hit.start_ms < end and hit.end_ms > start)
        if members:
            pieces.append(MergedInterval(start, end, members))
    return pieces


def build_intervals(
    hits: Sequence[Hit],
    *,
    merge_gap_ms: int,
    min_ms: int,
    max_ms: int,
    target_max_ms: int,
    video_duration_ms: int,
) -> list[MergedInterval]:
    """Full pipeline: merge, split the long ones, expand the short ones."""
    intervals: list[MergedInterval] = []
    for merged in merge_hits(hits, merge_gap_ms):
        for piece in split_long(merged, max_ms, target_max_ms):
            intervals.append(expand_to_minimum(piece, min_ms, video_duration_ms))
    return intervals
