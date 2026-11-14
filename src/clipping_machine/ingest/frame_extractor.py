"""Keyframe extraction and motion sampling in a single FFmpeg decode pass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from clipping_machine import ffmpeg
from clipping_machine.config import VideoConfig

KEYFRAME_PATTERN = "kf_%06d.jpg"


@dataclass(frozen=True)
class FrameExtraction:
    """Result of one pass over the video."""

    keyframes: tuple[Path, ...]
    keyframe_interval_ms: int
    motion: NDArray[np.float32]
    motion_fps: float

    def keyframe_time_ms(self, index: int) -> int:
        return index * self.keyframe_interval_ms

    def motion_between(self, start_ms: int, end_ms: int) -> float:
        """Mean frame-to-frame change (0–255 scale) inside ``[start_ms, end_ms)``."""
        if self.motion.size == 0:
            return 0.0
        first = int(start_ms / 1000 * self.motion_fps)
        last = max(first + 1, int(np.ceil(end_ms / 1000 * self.motion_fps)))
        chunk = self.motion[first:last]
        return float(chunk.mean()) if chunk.size else 0.0


def keyframe_path(directory: Path, index: int) -> Path:
    return directory / (KEYFRAME_PATTERN % index)


def motion_from_frames(frames: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Mean absolute difference between consecutive frames; first entry is 0."""
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    signed = frames.astype(np.int16)
    diffs = np.abs(np.diff(signed, axis=0)).mean(axis=(1, 2))
    return np.concatenate([np.zeros(1, dtype=np.float32), diffs.astype(np.float32)])


def extract_frames(
    video_path: Path,
    output_dir: Path,
    info: ffmpeg.MediaInfo,
    config: VideoConfig,
    on_progress: Callable[[float], None] | None = None,
) -> FrameExtraction:
    """Write one JPEG every ``keyframe_interval_seconds`` and sample motion.

    Keyframe ``k`` shows the video at ``k * keyframe_interval_seconds``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("kf_*.jpg"):
        stale.unlink()

    motion_width = config.motion_frame_width
    motion_height = ffmpeg.even(motion_width * info.height / max(info.width, 1))
    side = config.keyframe_max_side
    graph = (
        "[0:v]split=2[kin][min];"
        f"[kin]fps=1/{config.keyframe_interval_seconds:g},"
        f"scale=w='min({side},iw)':h='min({side},ih)':"
        "force_original_aspect_ratio=decrease:force_divisible_by=2[kf];"
        f"[min]fps={config.motion_sample_fps:g},scale={motion_width}:{motion_height},"
        "format=gray[gray]"
    )
    keyframe_output = [
        "-map",
        "[kf]",
        "-q:v",
        "4",
        "-start_number",
        "0",
        str(output_dir / KEYFRAME_PATTERN),
    ]

    expected = max(1, int(info.duration_ms / 1000 * config.motion_sample_fps))
    motion: list[float] = []
    previous: NDArray[np.int16] | None = None
    for count, frame in enumerate(
        ffmpeg.iter_gray_frames(
            video_path,
            fps=config.motion_sample_fps,
            width=motion_width,
            height=motion_height,
            filter_complex=graph,
            extra_outputs=keyframe_output,
        ),
        start=1,
    ):
        current = frame.astype(np.int16)
        motion.append(0.0 if previous is None else float(np.abs(current - previous).mean()))
        previous = current
        if on_progress is not None and count % 20 == 0:
            on_progress(min(1.0, count / expected))

    keyframes = tuple(sorted(output_dir.glob("kf_*.jpg")))
    if not keyframes:
        raise ffmpeg.FFmpegError("no keyframes could be extracted from the video")
    if on_progress is not None:
        on_progress(1.0)
    return FrameExtraction(
        keyframes=keyframes,
        keyframe_interval_ms=config.keyframe_interval_ms,
        motion=np.asarray(motion, dtype=np.float32),
        motion_fps=config.motion_sample_fps,
    )
