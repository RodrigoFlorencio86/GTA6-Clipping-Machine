"""Still-frame thumbnails framed the same way as the clip."""

from __future__ import annotations

from pathlib import Path

from clipping_machine import ffmpeg
from clipping_machine.clipping.vertical_formatter import Framing, vertical_filter


def generate_thumbnail(
    video_path: Path,
    info: ffmpeg.MediaInfo,
    time_ms: int,
    output_path: Path,
    *,
    width: int,
    height: int,
    framing: Framing = "center",
    focus_x: float | None = None,
) -> Path:
    """Write a JPEG of the frame at ``time_ms`` using the clip's vertical framing."""
    time_ms = min(max(0, time_ms), max(0, info.duration_ms - 100))
    chain = vertical_filter(
        info.width, info.height, width, height, framing=framing, focus_x=focus_x
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run_ffmpeg(
        [
            "-ss",
            ffmpeg.seconds(time_ms),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            chain,
            "-q:v",
            "2",
            "-update",
            "1",
            str(output_path),
        ]
    )
    return output_path
