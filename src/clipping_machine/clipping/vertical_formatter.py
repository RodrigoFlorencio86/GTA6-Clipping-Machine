"""Builds the FFmpeg filter chain that turns any source into a 9:16 frame.

Framings:

* ``center``   — scale to fill, crop around the middle.
* ``smart``    — same crop, but shifted towards where the motion is (see
  :func:`focus_from_frames`). Falls back to the centre when motion is diffuse.
* ``blur_pad`` — the whole frame fitted to the width over a blurred, zoomed copy;
  nothing is cropped away.

Before any of them, black borders baked into the recording (a 4:3 game inside a 16:9
capture, letterboxed cutscenes, an OBS canvas larger than the game) can be detected
and cut off, so they do not end up as black bars in the vertical clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from clipping_machine import ffmpeg

Framing = Literal["smart", "center", "blur_pad"]

_FOCUS_WIDTH = 96
_ANALYSIS_WIDTH = 320  # enough to locate a border to within ~6 px of a 1080p frame
_ANALYSIS_FPS = 4.0
# How far the crop may follow the action: 1.0 = fully, 0.0 = always centred.
_FOCUS_STRENGTH = 0.8
# Borders thinner than this fraction of the frame are ignored (encoder edge lines, noise).
_MIN_BORDER = 0.01
# Never keep less than this fraction of either dimension: something else is going on.
_MIN_KEEP = 0.5


@dataclass(frozen=True)
class ContentBox:
    """The part of the source frame that is not a black border, in source pixels."""

    x: int
    y: int
    width: int
    height: int

    def crop(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


@dataclass(frozen=True)
class FramingAnalysis:
    content_box: ContentBox | None = None
    focus_x: float | None = None


def detect_content_box(
    frames: NDArray[np.uint8], *, black_level: int = 24
) -> tuple[float, float, float, float] | None:
    """Area that is not a black border, as fractions ``(left, top, right, bottom)``.

    ``frames`` has shape ``(time, height, width)``. A row or column is border when its
    mean brightness stays at or below ``black_level`` in every frame (like FFmpeg's
    ``cropdetect`` accumulated over the clip). Returns ``None`` when there is nothing
    worth cropping, or when the result would be implausible (an all-black clip, or a
    box smaller than half the frame).
    """
    if frames.ndim != 3 or frames.shape[0] == 0:
        return None
    _, height, width = frames.shape
    if height < 4 or width < 4:
        return None
    rows = frames.mean(axis=2).max(axis=0)
    cols = frames.mean(axis=1).max(axis=0)
    live_rows = np.flatnonzero(rows > black_level)
    live_cols = np.flatnonzero(cols > black_level)
    if live_rows.size == 0 or live_cols.size == 0:
        return None

    def edges(live: NDArray[np.intp], size: int) -> tuple[int, int]:
        start, end = int(live[0]), int(live[-1]) + 1
        # Step one analysis pixel inwards past each real border: that pixel blends
        # border and picture after downscaling.
        start = 0 if start < _MIN_BORDER * size else start + 1
        end = size if size - end < _MIN_BORDER * size else end - 1
        return start, end

    top, bottom = edges(live_rows, height)
    left, right = edges(live_cols, width)
    if (top, bottom, left, right) == (0, height, 0, width):
        return None
    if bottom - top < _MIN_KEEP * height or right - left < _MIN_KEEP * width:
        return None
    return left / width, top / height, right / width, bottom / height


def content_box_from_fractions(
    fractions: tuple[float, float, float, float], width: int, height: int
) -> ContentBox:
    """Scale a fractional box to source pixels, rounding inwards to even values."""
    left, top, right, bottom = fractions

    def inward(low: float, high: float, size: int) -> tuple[int, int]:
        start = min(size, 2 * math.ceil(low * size / 2))
        end = max(start + 2, 2 * math.floor(high * size / 2))
        return start, min(size, end) - start

    x, w = inward(left, right, width)
    y, h = inward(top, bottom, height)
    return ContentBox(x, y, w, h)


def crop_filter(
    src_width: int, src_height: int, out_width: int, out_height: int, focus_x: float = 0.5
) -> str:
    """Scale-to-fill then crop. ``focus_x`` (0–1) is the horizontal centre to keep."""
    if src_width <= 0 or src_height <= 0:
        raise ValueError("source dimensions must be positive")
    focus = min(1.0, max(0.0, focus_x))
    if src_width * out_height >= src_height * out_width:
        # Source is wider than the target: fit the height, crop the sides.
        scaled_width = max(out_width, ffmpeg.even(src_width * out_height / src_height))
        max_x = scaled_width - out_width
        x = min(max_x, max(0, round(focus * scaled_width - out_width / 2)))
        return f"scale={scaled_width}:{out_height},crop={out_width}:{out_height}:{x}:0,setsar=1"
    # Source is narrower/taller than the target: fit the width, crop top and bottom.
    scaled_height = max(out_height, ffmpeg.even(src_height * out_width / src_width))
    y = (scaled_height - out_height) // 2
    return f"scale={out_width}:{scaled_height},crop={out_width}:{out_height}:0:{y},setsar=1"


def blur_pad_filter(out_width: int, out_height: int) -> str:
    """Whole frame over a blurred background."""
    return (
        "split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={out_width}:{out_height}:force_original_aspect_ratio=increase,"
        f"crop={out_width}:{out_height},boxblur=24:2[bg];"
        f"[fgsrc]scale={out_width}:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def vertical_filter(
    src_width: int,
    src_height: int,
    out_width: int,
    out_height: int,
    *,
    framing: Framing = "center",
    focus_x: float | None = None,
    src_fps: float = 0.0,
    max_fps: float | None = None,
    content_box: ContentBox | None = None,
) -> str:
    """Complete ``-vf`` chain for one clip.

    With a ``content_box``, the black borders are cut first and the framing works on
    what is left (``focus_x`` is then relative to the box).
    """
    prefix = ""
    if content_box is not None:
        prefix = content_box.crop() + ","
        src_width, src_height = content_box.width, content_box.height
    if framing == "blur_pad":
        chain = prefix + blur_pad_filter(out_width, out_height)
    else:
        focus = focus_x if (framing == "smart" and focus_x is not None) else 0.5
        chain = prefix + crop_filter(src_width, src_height, out_width, out_height, focus)
    if max_fps is not None and src_fps > max_fps + 0.01:
        chain += f",fps={max_fps:g}"
    return chain


def focus_from_frames(frames: NDArray[np.uint8]) -> float:
    """Horizontal centre of localized motion, as a fraction of the frame width.

    ``frames`` has shape ``(time, height, width)``. Motion common to the whole frame
    (camera pans) is removed by subtracting the median column energy, so only
    localized action pulls the crop. Returns 0.5 when there is no clear focus.
    """
    if frames.ndim != 3 or frames.shape[0] < 2 or frames.shape[2] < 2:
        return 0.5
    energy = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(0, 1))
    localized = np.clip(energy - np.median(energy), 0.0, None)
    total = float(localized.sum())
    if total < 1e-3 * frames.shape[2]:
        return 0.5
    positions = np.arange(frames.shape[2], dtype=np.float64)
    centroid = float((positions * localized).sum() / total) / (frames.shape[2] - 1)
    return float(0.5 + _FOCUS_STRENGTH * (centroid - 0.5))


def analyze_framing(
    video_path: Path,
    info: ffmpeg.MediaInfo,
    start_ms: int,
    end_ms: int,
    *,
    detect_borders: bool,
    track_focus: bool,
    black_level: int = 24,
) -> FramingAnalysis:
    """One low-resolution pass over the clip: black borders and/or where the action is."""
    if not (detect_borders or track_focus) or info.width <= 0 or info.height <= 0:
        return FramingAnalysis()
    if not detect_borders:
        return FramingAnalysis(focus_x=estimate_focus_x(video_path, info, start_ms, end_ms))
    width = ffmpeg.even(min(_ANALYSIS_WIDTH, info.width))
    height = ffmpeg.even(width * info.height / info.width)
    frames = list(
        ffmpeg.iter_gray_frames(
            video_path,
            fps=_ANALYSIS_FPS,
            width=width,
            height=height,
            start_ms=start_ms,
            duration_ms=max(1, end_ms - start_ms),
        )
    )
    if not frames:
        return FramingAnalysis()
    stack = np.stack(frames)
    fractions = detect_content_box(stack, black_level=black_level)
    box = None
    if fractions is not None:
        box = content_box_from_fractions(fractions, info.width, info.height)
        left, top, right, bottom = fractions
        rows = slice(round(top * height), round(bottom * height))
        cols = slice(round(left * width), round(right * width))
        stack = stack[:, rows, cols]
    focus = focus_from_frames(stack) if track_focus else None
    return FramingAnalysis(content_box=box, focus_x=focus)


def estimate_focus_x(video_path: Path, info: ffmpeg.MediaInfo, start_ms: int, end_ms: int) -> float:
    """Sample the clip range at low resolution and locate the action horizontally."""
    height = ffmpeg.even(_FOCUS_WIDTH * info.height / max(info.width, 1))
    frames = list(
        ffmpeg.iter_gray_frames(
            video_path,
            fps=_ANALYSIS_FPS,
            width=_FOCUS_WIDTH,
            height=height,
            start_ms=start_ms,
            duration_ms=max(1, end_ms - start_ms),
        )
    )
    if len(frames) < 2:
        return 0.5
    return focus_from_frames(np.stack(frames))
