"""Thin, typed wrappers around the ``ffmpeg`` / ``ffprobe`` binaries.

Every call passes an argument list (never a shell string), so file names cannot
inject options or commands.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ProgressCallback = Callable[[float], None]

_BASE_ARGS = ("-hide_banner", "-nostdin", "-loglevel", "error")


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe exited with an error."""


class FFmpegNotFoundError(FFmpegError):
    """The ffmpeg/ffprobe binary is not installed or not on PATH."""


@dataclass(frozen=True)
class MediaInfo:
    """Subset of ffprobe output the app cares about."""

    duration_ms: int
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    size_bytes: int
    format_name: str

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


def _binary(name: str, env_var: str) -> str:
    explicit = os.environ.get(env_var)
    if explicit:
        return explicit
    found = shutil.which(name)
    if not found:
        raise FFmpegNotFoundError(
            f"'{name}' not found. Install FFmpeg and add it to PATH "
            f"(or point {env_var} at the executable)."
        )
    return found


def ffmpeg_bin() -> str:
    return _binary("ffmpeg", "GCM_FFMPEG")


def ffprobe_bin() -> str:
    return _binary("ffprobe", "GCM_FFPROBE")


def is_available() -> bool:
    try:
        ffmpeg_bin()
        ffprobe_bin()
    except FFmpegNotFoundError:
        return False
    return True


def _tail(text: str, limit: int = 1200) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def _parse_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_duration(*candidates: Any) -> float:
    for candidate in candidates:
        try:
            seconds = float(candidate)
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return seconds
    return 0.0


def probe(path: Path) -> MediaInfo:
    """Read container/stream metadata. Raises :class:`FFmpegError` if unreadable."""
    command = [
        ffprobe_bin(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe could not read the file: {_tail(result.stderr)}")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError("ffprobe returned an invalid response") from exc

    streams: list[dict[str, Any]] = payload.get("streams") or []
    container: dict[str, Any] = payload.get("format") or {}
    video = next(
        (
            stream
            for stream in streams
            if stream.get("codec_type") == "video"
            and not (stream.get("disposition") or {}).get("attached_pic")
        ),
        None,
    )
    if video is None:
        raise FFmpegError("the file has no video stream")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)

    duration = _parse_duration(container.get("duration"), video.get("duration"))
    fps = _parse_rate(video.get("avg_frame_rate")) or _parse_rate(video.get("r_frame_rate"))
    try:
        size_bytes = int(container.get("size") or path.stat().st_size)
    except (OSError, ValueError):
        size_bytes = 0

    return MediaInfo(
        duration_ms=round(duration * 1000),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        video_codec=str(video.get("codec_name") or "unknown"),
        audio_codec=str(audio.get("codec_name") or "unknown") if audio else None,
        size_bytes=size_bytes,
        format_name=str(container.get("format_name") or "unknown"),
    )


def run_ffmpeg(
    args: Sequence[str],
    *,
    duration_ms: int | None = None,
    on_progress: ProgressCallback | None = None,
    cwd: Path | None = None,
) -> None:
    """Run ffmpeg with ``args`` (everything after the binary name).

    When ``on_progress`` and ``duration_ms`` are given, progress in ``[0, 1]`` is
    reported from ffmpeg's ``-progress`` stream.
    """
    command = [ffmpeg_bin(), *_BASE_ARGS, "-y"]
    track = on_progress is not None and bool(duration_ms)
    if track:
        command += ["-progress", "pipe:1", "-nostats"]
    command += list(args)

    with tempfile.TemporaryFile(mode="w+b") as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE if track else subprocess.DEVNULL,
            stderr=stderr_file,
            cwd=str(cwd) if cwd else None,
        )
        if track and process.stdout is not None and on_progress and duration_ms:
            for raw in process.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                # ffmpeg reports microseconds under both names.
                if line.startswith(("out_time_us=", "out_time_ms=")):
                    try:
                        micros = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    on_progress(max(0.0, min(1.0, micros / 1000 / duration_ms)))
        returncode = process.wait()
        if returncode != 0:
            stderr_file.seek(0)
            message = stderr_file.read().decode("utf-8", errors="replace")
            raise FFmpegError(f"ffmpeg failed (exit code {returncode}): {_tail(message)}")
    if on_progress is not None:
        on_progress(1.0)


def even(value: float) -> int:
    """Round to the nearest even integer >= 2 (codecs need even dimensions)."""
    return max(2, round(value / 2) * 2)


def seconds(ms: int) -> str:
    """Milliseconds → ffmpeg time argument."""
    return f"{ms / 1000:.3f}"


def iter_gray_frames(
    path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    start_ms: int | None = None,
    duration_ms: int | None = None,
    extra_outputs: Sequence[str] = (),
    filter_complex: str | None = None,
) -> Iterator[NDArray[np.uint8]]:
    """Yield low-resolution grayscale frames sampled at ``fps``.

    ``filter_complex`` / ``extra_outputs`` let a caller piggyback extra outputs
    (e.g. keyframe JPEGs) on the same decode pass; the graph must then expose a
    ``[gray]`` pad producing ``width``x``height`` gray frames.
    """
    command: list[str] = [ffmpeg_bin(), *_BASE_ARGS, "-y"]
    if start_ms is not None:
        command += ["-ss", seconds(start_ms)]
    command += ["-i", str(path)]
    if duration_ms is not None:
        command += ["-t", seconds(duration_ms)]
    if filter_complex is None:
        command += ["-vf", f"fps={fps},scale={width}:{height},format=gray", "-an"]
    else:
        command += ["-filter_complex", filter_complex, *extra_outputs, "-map", "[gray]"]
    command += ["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]

    frame_bytes = width * height
    with tempfile.TemporaryFile(mode="w+b") as stderr_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)
        assert process.stdout is not None
        try:
            while True:
                buffer = process.stdout.read(frame_bytes)
                if len(buffer) < frame_bytes:
                    break
                yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width)
        finally:
            process.stdout.close()
            returncode = process.wait()
        if returncode != 0:
            stderr_file.seek(0)
            message = stderr_file.read().decode("utf-8", errors="replace")
            raise FFmpegError(f"ffmpeg failed (exit code {returncode}): {_tail(message)}")
