"""Registers a local video file in the library.

Two entry points, both strictly local:

* :meth:`VideoImporter.import_upload` — bytes received from the browser upload widget.
* :meth:`VideoImporter.import_path` — a file that already lives on this machine
  (recommended for multi-gigabyte recordings; it can be indexed in place, with no copy).
"""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO

from clipping_machine.config import Settings
from clipping_machine.ingest.video_validator import (
    ValidationIssue,
    VideoValidationError,
    validate_source,
    validate_video,
)
from clipping_machine.storage.database import utc_now
from clipping_machine.storage.models import Video, VideoStatus
from clipping_machine.storage.repositories import VideoRepository

_FINGERPRINT_CHUNK = 1024 * 1024
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class RightsNotConfirmedError(PermissionError):
    """The user did not confirm they own / are allowed to use the footage."""


class DuplicateVideoError(ValueError):
    """The same file is already in the library."""

    def __init__(self, existing: Video) -> None:
        self.existing = existing
        super().__init__(f"This video is already in the library as {existing.video_id}.")


def new_video_id() -> str:
    return f"vid_{uuid.uuid4().hex[:12]}"


def safe_filename(name: str) -> str:
    """Strip directories and unusual characters from a user-supplied file name."""
    base = Path(name.replace("\\", "/")).name
    cleaned = _SAFE_NAME.sub("_", base).strip("._")
    return cleaned or "video"


def fingerprint(path: Path) -> str:
    """Cheap content fingerprint: size + first and last megabyte (fast on huge files)."""
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(_FINGERPRINT_CHUNK))
        if size > 2 * _FINGERPRINT_CHUNK:
            handle.seek(size - _FINGERPRINT_CHUNK)
            digest.update(handle.read(_FINGERPRINT_CHUNK))
    return digest.hexdigest()


class VideoImporter:
    def __init__(self, settings: Settings, videos: VideoRepository) -> None:
        self._settings = settings
        self._videos = videos

    def import_upload(
        self, stream: BinaryIO, original_name: str, *, rights_confirmed: bool
    ) -> Video:
        """Persist an uploaded stream under ``data/uploads`` and register it."""
        self._require_rights(rights_confirmed)
        video_id = new_video_id()
        name = safe_filename(original_name)
        extension = Path(name).suffix.lower()
        if extension not in self._settings.upload.allowed_extensions:
            allowed = ", ".join(self._settings.upload.allowed_extensions)
            raise VideoValidationError(
                (
                    ValidationIssue(
                        "unsupported_extension",
                        f"Unsupported file type '{extension or '(none)'}'. Allowed: {allowed}.",
                    ),
                )
            )
        self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        target = self._settings.uploads_dir / f"{video_id}{extension}"
        try:
            with target.open("wb") as handle:
                shutil.copyfileobj(stream, handle, length=8 * 1024 * 1024)
            return self._register(target, video_id, original_name=name, managed_copy=True)
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def import_path(self, source: str, *, rights_confirmed: bool, copy: bool = False) -> Video:
        """Register a file already on disk. With ``copy=False`` it is indexed in place."""
        self._require_rights(rights_confirmed)
        issues = validate_source(source, self._settings.upload)
        if issues:
            raise VideoValidationError(issues)
        path = Path(source.strip().strip('"')).resolve()
        video_id = new_video_id()
        if not copy:
            return self._register(path, video_id, original_name=path.name, managed_copy=False)

        self._settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        target = self._settings.uploads_dir / f"{video_id}{path.suffix.lower()}"
        try:
            shutil.copyfile(path, target)
            return self._register(target, video_id, original_name=path.name, managed_copy=True)
        except Exception:
            target.unlink(missing_ok=True)
            raise

    @staticmethod
    def _require_rights(rights_confirmed: bool) -> None:
        if not rights_confirmed:
            raise RightsNotConfirmedError(
                "Confirm that this is your own gameplay (or that you are authorized to use it) "
                "before importing."
            )

    def _register(
        self, path: Path, video_id: str, *, original_name: str, managed_copy: bool
    ) -> Video:
        result = validate_video(path, self._settings.upload)
        if not result.ok or result.info is None:
            raise VideoValidationError(result.issues)

        print_ = fingerprint(path)
        existing = self._videos.find_by_fingerprint(print_)
        if existing is not None:
            raise DuplicateVideoError(existing)

        info = result.info
        now = utc_now()
        video = Video(
            video_id=video_id,
            original_name=original_name,
            path=str(path),
            managed_copy=managed_copy,
            size_bytes=path.stat().st_size,
            fingerprint=print_,
            duration_ms=info.duration_ms,
            width=info.width,
            height=info.height,
            fps=info.fps,
            video_codec=info.video_codec,
            audio_codec=info.audio_codec,
            rights_confirmed_at=now,
            created_at=now,
            status=VideoStatus.UPLOADED,
        )
        self._videos.add(video)
        return video
