"""Sliding temporal windows over a video."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    index: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def mid_ms(self) -> int:
        return (self.start_ms + self.end_ms) // 2


def compute_windows(duration_ms: int, window_ms: int, overlap_ms: int) -> list[Window]:
    """Split ``[0, duration_ms)`` into windows of ``window_ms`` overlapping by ``overlap_ms``.

    The last window is clipped to the video end. A trailing window that would only
    repeat footage already covered by the previous one is dropped.
    """
    if duration_ms <= 0:
        return []
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if not 0 <= overlap_ms < window_ms:
        raise ValueError("overlap_ms must be in [0, window_ms)")

    stride = window_ms - overlap_ms
    windows: list[Window] = []
    start = 0
    while start < duration_ms:
        end = min(start + window_ms, duration_ms)
        windows.append(Window(len(windows), start, end))
        if end >= duration_ms:
            break
        start += stride
    return windows


def keyframe_times_ms(duration_ms: int, interval_ms: int) -> list[int]:
    """Timestamps of the keyframe grid: 0, interval, 2*interval, ... (< duration)."""
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return list(range(0, max(duration_ms, 1), interval_ms))


def keyframes_for_window(window: Window, interval_ms: int, total_keyframes: int) -> list[int]:
    """Indices of the grid keyframes that fall inside ``window``.

    Always returns at least one index (the nearest keyframe) when any keyframe exists.
    """
    if total_keyframes <= 0:
        return []
    first = -(-window.start_ms // interval_ms)  # ceil division
    indices = [
        index for index in range(first, total_keyframes) if index * interval_ms < window.end_ms
    ]
    if indices:
        return indices
    nearest = min(round(window.mid_ms / interval_ms), total_keyframes - 1)
    return [max(0, nearest)]
