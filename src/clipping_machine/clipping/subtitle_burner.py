"""SRT generation and the FFmpeg filter that burns subtitles into the picture."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SUBTITLE_FILENAME = "subtitles.srt"
_STYLE = "Fontsize=13,Bold=1,Alignment=2,MarginV=48,Outline=2,Shadow=0,BorderStyle=1"


@dataclass(frozen=True)
class Cue:
    start_ms: int
    end_ms: int
    text: str


def format_timestamp(ms: int) -> str:
    ms = max(0, int(ms))
    hours, remainder = divmod(ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def wrap_text(text: str, max_chars: int = 28, max_lines: int = 2) -> str:
    """Short lines read better on a vertical frame."""
    lines = textwrap.wrap(" ".join(text.split()), width=max_chars) or [""]
    if len(lines) > max_lines:
        lines = [*lines[: max_lines - 1], " ".join(lines[max_lines - 1 :])]
    return "\n".join(lines)


def format_srt(cues: Sequence[Cue]) -> str:
    blocks: list[str] = []
    for cue in sorted(cues, key=lambda item: item.start_ms):
        text = wrap_text(cue.text)
        if not text.strip() or cue.end_ms <= cue.start_ms:
            continue
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{format_timestamp(cue.start_ms)} --> {format_timestamp(cue.end_ms)}\n"
            f"{text}\n"
        )
    return "\n".join(blocks)


def write_srt(cues: Sequence[Cue], directory: Path) -> Path | None:
    """Write ``subtitles.srt`` into ``directory``; returns None when there is nothing to show."""
    content = format_srt(cues)
    if not content:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SUBTITLE_FILENAME
    path.write_text(content, encoding="utf-8")
    return path


def subtitles_filter(filename: str = SUBTITLE_FILENAME) -> str:
    """Filter fragment. FFmpeg must run with the SRT's folder as working directory:
    a bare file name sidesteps the drive-letter/colon escaping rules of filtergraphs.
    """
    if any(char in filename for char in "\\/:'[],;"):
        raise ValueError("subtitle file name must be a plain name without separators")
    return f"subtitles=filename={filename}:charenc=UTF-8:force_style='{_STYLE}'"
