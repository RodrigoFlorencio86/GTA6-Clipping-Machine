"""Validation of user-supplied video files.

Only files that already exist on this machine are accepted. Anything that looks
like a URL is rejected on purpose: this project does not download content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from clipping_machine import ffmpeg
from clipping_machine.config import UploadConfig

_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]{1,}://")
_URL_HINTS = ("www.", "youtu.be/", "youtube.com/", "tiktok.com/", "twitch.tv/", "instagram.com/")


@dataclass(frozen=True)
class ValidationIssue:
    """A machine-readable reason (``code``) plus an English message."""

    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...] = ()
    info: ffmpeg.MediaInfo | None = field(default=None)

    @property
    def ok(self) -> bool:
        return not self.issues and self.info is not None


class VideoValidationError(ValueError):
    """Raised when a file fails validation."""

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues))


def looks_like_url(value: str) -> bool:
    """True for ``scheme://`` strings and well-known platform hosts.

    Windows drive paths (``C:\\...``) are not URLs: the scheme regex needs ``://``
    after at least two characters.
    """
    text = value.strip()
    if _URL_SCHEME.match(text):
        return True
    lowered = text.lower()
    return any(lowered.startswith(hint) or f"/{hint}" in lowered for hint in _URL_HINTS)


def validate_source(value: str, config: UploadConfig) -> tuple[ValidationIssue, ...]:
    """Cheap checks that do not need ffprobe (URL, existence, extension, size)."""
    if looks_like_url(value):
        return (
            ValidationIssue(
                "url_not_supported",
                "Links are not supported. Only video files stored on this computer are accepted.",
            ),
        )
    path = Path(value.strip().strip('"'))
    if not path.is_file():
        return (ValidationIssue("not_found", f"File not found: {path}"),)

    issues: list[ValidationIssue] = []
    extension = path.suffix.lower()
    if extension not in config.allowed_extensions:
        allowed = ", ".join(config.allowed_extensions)
        issues.append(
            ValidationIssue(
                "unsupported_extension",
                f"Unsupported file type '{extension or '(none)'}'. Allowed: {allowed}.",
            )
        )
    size = path.stat().st_size
    if size == 0:
        issues.append(ValidationIssue("empty_file", "The file is empty."))
    elif size > config.max_file_size_bytes:
        issues.append(
            ValidationIssue(
                "file_too_large",
                f"File is {size / 1024**2:.0f} MB; the configured limit is "
                f"{config.max_file_size_mb} MB.",
            )
        )
    return tuple(issues)


def validate_video(path: Path, config: UploadConfig) -> ValidationResult:
    """Full validation: file checks, then a real ffprobe read of the streams."""
    issues = list(validate_source(str(path), config))
    if issues:
        return ValidationResult(tuple(issues))

    try:
        info = ffmpeg.probe(path)
    except ffmpeg.FFmpegNotFoundError:
        raise
    except ffmpeg.FFmpegError as exc:
        return ValidationResult((ValidationIssue("unreadable", f"Not a readable video: {exc}"),))

    if info.duration_ms <= 0:
        issues.append(ValidationIssue("no_duration", "Could not determine the video duration."))
    elif info.duration_ms > config.max_duration_ms:
        issues.append(
            ValidationIssue(
                "too_long",
                f"Video is {info.duration_ms / 60000:.1f} min; the configured limit is "
                f"{config.max_video_duration_minutes:g} min.",
            )
        )
    if info.width < config.min_width or info.height < config.min_height:
        issues.append(
            ValidationIssue(
                "resolution_too_low",
                f"Resolution {info.width}x{info.height} is below the minimum "
                f"{config.min_width}x{config.min_height}.",
            )
        )
    return ValidationResult(tuple(issues), info)
