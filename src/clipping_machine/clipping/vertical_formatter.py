"""Builds the FFmpeg filter chain that turns any source into a 9:16 frame.

Framings:

* ``center``   — scale to fill, crop around the middle.
* ``smart``    — same crop, but shifted towards where the motion is (see
  :func:`focus_from_frames`). Falls back to the centre when motion is diffuse.
* ``blur_pad`` — the whole frame fitted to the width over a blurred, zoomed copy;
  nothing is cropped away.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from clipping_machine import ffmpeg

Framing = Literal["smart", "center", "blur_pad"]

_FOCUS_WIDTH = 96
_FOCUS_FPS = 4.0
# How far the crop may follow the action: 1.0 = fully, 0.0 = always centred.
_FOCUS_STRENGTH = 0.8


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
) -> str:
    """Complete ``-vf`` chain for one clip."""
    if framing == "blur_pad":
        chain = blur_pad_filter(out_width, out_height)
    else:
        focus = focus_x if (framing == "smart" and focus_x is not None) else 0.5
        chain = crop_filter(src_width, src_height, out_width, out_height, focus)
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


def estimate_focus_x(video_path: Path, info: ffmpeg.MediaInfo, start_ms: int, end_ms: int) -> float:
    """Sample the clip range at low resolution and locate the action horizontally."""
    height = ffmpeg.even(_FOCUS_WIDTH * info.height / max(info.width, 1))
    frames = list(
        ffmpeg.iter_gray_frames(
            video_path,
            fps=_FOCUS_FPS,
            width=_FOCUS_WIDTH,
            height=height,
            start_ms=start_ms,
            duration_ms=max(1, end_ms - start_ms),
        )
    )
    if len(frames) < 2:
        return 0.5
    return focus_from_frames(np.stack(frames))
