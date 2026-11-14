"""Cuts a time range out of the source and encodes it as a vertical H.264 MP4."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from clipping_machine import ffmpeg
from clipping_machine.clipping.subtitle_burner import subtitles_filter
from clipping_machine.clipping.vertical_formatter import ContentBox, Framing, vertical_filter
from clipping_machine.config import ExportConfig


@dataclass(frozen=True)
class RenderOptions:
    width: int
    height: int
    crf: int
    preset: str
    audio_bitrate: str
    max_fps: float
    framing: Framing = "center"
    focus_x: float | None = None
    content_box: ContentBox | None = None

    @classmethod
    def final(
        cls,
        config: ExportConfig,
        *,
        framing: Framing | None = None,
        focus_x: float | None = None,
        content_box: ContentBox | None = None,
    ) -> RenderOptions:
        return cls(
            width=config.width,
            height=config.height,
            crf=config.crf,
            preset=config.preset,
            audio_bitrate=config.audio_bitrate,
            max_fps=config.max_fps,
            framing=framing or config.framing,
            focus_x=focus_x,
            content_box=content_box,
        )

    @classmethod
    def draft(
        cls,
        config: ExportConfig,
        *,
        framing: Framing | None = None,
        focus_x: float | None = None,
        content_box: ContentBox | None = None,
    ) -> RenderOptions:
        """Small, fast encode used for in-app review."""
        height = ffmpeg.even(config.draft_height)
        width = ffmpeg.even(height * config.width / config.height)
        return cls(
            width=width,
            height=height,
            crf=30,
            preset="ultrafast",
            audio_bitrate="96k",
            max_fps=30.0,
            framing=framing or config.framing,
            focus_x=focus_x,
            content_box=content_box,
        )


def format_position(ms: int) -> str:
    """Position inside the source video as ``HH-MM-SS`` (file-name safe)."""
    total = max(0, ms) // 1000
    return f"{total // 3600:02d}-{total % 3600 // 60:02d}-{total % 60:02d}"


def clip_filename(prefix: str, day: date, start_ms: int, score: float) -> str:
    """``gta6_clip_2026-11-14_00-12-04_score093.mp4``: date, start position, score."""
    score_part = f"{round(min(1.0, max(0.0, score)) * 100):03d}"
    return f"{prefix}_{day.isoformat()}_{format_position(start_ms)}_score{score_part}.mp4"


def unique_path(path: Path) -> Path:
    """Append ``_2``, ``_3``... rather than overwrite an existing export."""
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def render_clip(
    video_path: Path,
    info: ffmpeg.MediaInfo,
    start_ms: int,
    end_ms: int,
    output_path: Path,
    options: RenderOptions,
    *,
    subtitles_path: Path | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Encode ``[start_ms, end_ms)`` to ``output_path`` (MP4, H.264 + AAC, yuv420p)."""
    if end_ms <= start_ms:
        raise ValueError("clip end must be after its start")
    if start_ms < 0 or start_ms >= info.duration_ms:
        raise ValueError("clip start is outside the video")
    end_ms = min(end_ms, info.duration_ms)
    duration_ms = end_ms - start_ms

    chain = vertical_filter(
        info.width,
        info.height,
        options.width,
        options.height,
        framing=options.framing,
        focus_x=options.focus_x,
        src_fps=info.fps,
        max_fps=options.max_fps,
        content_box=options.content_box,
    )
    cwd: Path | None = None
    if subtitles_path is not None:
        chain += "," + subtitles_filter(subtitles_path.name)
        cwd = subtitles_path.parent.resolve()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-ss",
        ffmpeg.seconds(start_ms),
        "-i",
        str(video_path.resolve()),
        "-t",
        ffmpeg.seconds(duration_ms),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-sn",
        "-dn",
        "-vf",
        chain,
        "-c:v",
        "libx264",
        "-preset",
        options.preset,
        "-crf",
        str(options.crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        options.audio_bitrate,
        "-ar",
        "48000",
        "-ac",
        "2",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    ffmpeg.run_ffmpeg(args, duration_ms=duration_ms, on_progress=on_progress, cwd=cwd)
    return output_path
