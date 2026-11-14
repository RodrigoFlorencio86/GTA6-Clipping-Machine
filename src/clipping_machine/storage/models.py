"""Domain records persisted in SQLite."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class VideoStatus(StrEnum):
    UPLOADED = "uploaded"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class ClipStatus(StrEnum):
    CANDIDATE = "candidate"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPORTED = "exported"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass
class Video:
    video_id: str
    original_name: str
    path: str
    managed_copy: bool
    size_bytes: int
    fingerprint: str
    duration_ms: int
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    rights_confirmed_at: str
    created_at: str
    status: VideoStatus = VideoStatus.UPLOADED
    stage: str = ""
    progress: float = 0.0
    error: str | None = None
    segments_count: int = 0
    indexed_at: str | None = None
    index_seconds: float | None = None
    embedding_model: str | None = None

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None


@dataclass(frozen=True)
class Scene:
    video_id: str
    scene_index: int
    start_ms: int
    end_ms: int


@dataclass
class Segment:
    segment_id: str
    video_id: str
    seg_index: int
    start_ms: int
    end_ms: int
    preview_frame_path: str
    embedding_model: str
    embedding_dimension: int
    audio_energy: float = 0.0
    audio_db: float = -80.0
    motion: float = 0.0
    motion_raw: float = 0.0
    scene_cuts: int = 0
    reaction: float = 0.0
    tags: list[str] = field(default_factory=list)
    transcript: str = ""
    status: str = "indexed"

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def visual_description(self) -> str:
        return ", ".join(self.tags)


@dataclass
class Clip:
    clip_id: str
    video_id: str
    start_ms: int
    end_ms: int
    query: str
    title: str
    score: float
    score_breakdown: dict[str, float]
    reason: str
    created_at: str
    updated_at: str
    status: ClipStatus = ClipStatus.CANDIDATE
    draft_path: str | None = None
    output_path: str | None = None
    thumbnail_path: str | None = None
    review_note: str = ""
    exported_at: str | None = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass(frozen=True)
class ClipEvent:
    clip_id: str
    from_status: ClipStatus | None
    to_status: ClipStatus
    note: str
    created_at: str
