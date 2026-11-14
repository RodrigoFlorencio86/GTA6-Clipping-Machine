"""Typed configuration loaded from ``configs/*.yaml`` plus a few environment overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "configs"
SUPPORTED_DIMENSIONS = (128, 256, 512, 768)
SUPPORTED_VISION_TOKENS = (70, 140, 280, 560, 1120)


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UploadConfig(_Section):
    allowed_extensions: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".webm")
    max_file_size_mb: int = Field(default=20480, gt=0)
    max_video_duration_minutes: float = Field(default=180, gt=0)
    min_width: int = Field(default=320, gt=0)
    min_height: int = Field(default=180, gt=0)

    @field_validator("allowed_extensions")
    @classmethod
    def _normalize_extensions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in value)

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def max_duration_ms(self) -> int:
        return int(self.max_video_duration_minutes * 60_000)


class VideoConfig(_Section):
    keyframe_interval_seconds: float = Field(default=2.0, gt=0)
    window_duration_seconds: float = Field(default=8.0, gt=0)
    window_overlap_seconds: float = Field(default=2.0, ge=0)
    audio_sample_rate: int = Field(default=16000, gt=0)
    keyframe_max_side: int = Field(default=768, ge=64)
    motion_sample_fps: float = Field(default=4.0, gt=0)
    motion_frame_width: int = Field(default=160, ge=16)

    @model_validator(mode="after")
    def _overlap_shorter_than_window(self) -> VideoConfig:
        if self.window_overlap_seconds >= self.window_duration_seconds:
            raise ValueError("window_overlap_seconds must be smaller than window_duration_seconds")
        return self

    @property
    def window_ms(self) -> int:
        return int(self.window_duration_seconds * 1000)

    @property
    def overlap_ms(self) -> int:
        return int(self.window_overlap_seconds * 1000)

    @property
    def keyframe_interval_ms(self) -> int:
        return int(self.keyframe_interval_seconds * 1000)


class SceneConfig(_Section):
    enabled: bool = True
    threshold: float = Field(default=27.0, gt=0)
    min_scene_seconds: float = Field(default=1.0, gt=0)
    frame_skip: int = Field(default=1, ge=0)
    snap_tolerance_seconds: float = Field(default=1.0, ge=0)


class EmbeddingConfig(_Section):
    model: str = "google/embeddinggemma-2"
    dimension: int = 768
    reduced_dimensions: tuple[int, ...] = (128, 256, 512)
    normalize: bool = True
    device: str = "auto"
    dtype: Literal["auto", "float32", "bfloat16"] = "auto"
    batch_size: int = Field(default=4, gt=0)
    include_audio: bool = True
    local_files_only: bool = False
    query_prompt_name: str = "SearchQuery"
    # Soft tokens per video frame. "auto" = 140 on GPU, 70 on CPU (about 2x faster).
    vision_tokens: Literal["auto"] | int = "auto"

    @field_validator("vision_tokens")
    @classmethod
    def _supported_vision_tokens(cls, value: str | int) -> str | int:
        if value != "auto" and value not in SUPPORTED_VISION_TOKENS:
            raise ValueError(f"vision_tokens must be 'auto' or one of {SUPPORTED_VISION_TOKENS}")
        return value

    @field_validator("dimension")
    @classmethod
    def _supported_dimension(cls, value: int) -> int:
        if value not in SUPPORTED_DIMENSIONS:
            raise ValueError(f"dimension must be one of {SUPPORTED_DIMENSIONS}")
        return value


class CalibrationConfig(_Section):
    """Rescales cosine similarity for the *editorial score* only.

    With EmbeddingGemma 2, text↔video cosine sits in a narrow band: about 0.5–0.6
    for unrelated content and 0.7–0.8 for a clear match (docs/MODELS.md). Feeding
    that straight into the score would make the semantic term nearly constant, so
    the score uses ``(cosine - floor) / (ceiling - floor)`` clipped to 0–1.
    ``min_similarity`` is always compared against the raw cosine.
    """

    enabled: bool = True
    floor: float = 0.55
    ceiling: float = 0.80

    @model_validator(mode="after")
    def _ordered(self) -> CalibrationConfig:
        if self.ceiling <= self.floor:
            raise ValueError("calibration ceiling must be greater than floor")
        return self


class SearchConfig(_Section):
    top_k: int = Field(default=50, gt=0)
    min_similarity: float = Field(default=0.65, ge=0, le=1)
    # A moment must also be within this distance of the query's best similarity, so loud or
    # busy footage that matches the query only loosely cannot outrank the real match.
    # 0 disables it (only ``min_similarity`` applies).
    relative_margin: float = Field(default=0.04, ge=0, le=1)
    merge_gap_seconds: float = Field(default=3.0, ge=0)
    min_clip_duration_seconds: float = Field(default=8.0, gt=0)
    max_clip_duration_seconds: float = Field(default=60.0, gt=0)
    target_clip_duration_seconds: tuple[float, float] = (15.0, 45.0)
    min_results: int = Field(default=10, ge=0)
    calibration: CalibrationConfig = CalibrationConfig()

    @model_validator(mode="after")
    def _durations_ordered(self) -> SearchConfig:
        low, high = self.target_clip_duration_seconds
        if not self.min_clip_duration_seconds <= low <= high <= self.max_clip_duration_seconds:
            raise ValueError("expected min_clip <= target_min <= target_max <= max_clip duration")
        return self

    @property
    def merge_gap_ms(self) -> int:
        return int(self.merge_gap_seconds * 1000)

    @property
    def min_clip_ms(self) -> int:
        return int(self.min_clip_duration_seconds * 1000)

    @property
    def max_clip_ms(self) -> int:
        return int(self.max_clip_duration_seconds * 1000)

    @property
    def target_ms(self) -> tuple[int, int]:
        low, high = self.target_clip_duration_seconds
        return int(low * 1000), int(high * 1000)


class ScoreWeights(_Section):
    semantic_similarity: float = 0.35
    audio_intensity: float = 0.20
    visual_motion: float = 0.15
    reaction_presence: float = 0.15
    novelty: float = 0.10
    duration_fit: float = 0.05

    @model_validator(mode="after")
    def _sum_to_one(self) -> ScoreWeights:
        total = sum(self.model_dump().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"score weights must sum to 1.0, got {total:.4f}")
        if any(weight < 0 for weight in self.model_dump().values()):
            raise ValueError("score weights must be non-negative")
        return self


class ExportConfig(_Section):
    width: int = Field(default=1080, gt=0)
    height: int = Field(default=1920, gt=0)
    max_fps: float = Field(default=60.0, gt=0)
    crf: int = Field(default=20, ge=0, le=51)
    preset: str = "medium"
    audio_bitrate: str = "160k"
    framing: Literal["smart", "center", "blur_pad"] = "smart"
    filename_prefix: str = "gta6_clip"
    draft_height: int = Field(default=640, ge=160)
    # Cut black borders recorded around the picture (4:3 games, letterboxed cutscenes,
    # an OBS canvas larger than the game) before framing. No-op when there are none.
    remove_black_borders: bool = True
    black_level: int = Field(default=24, ge=0, le=255)  # mean luma at or below = border

    @model_validator(mode="after")
    def _vertical(self) -> ExportConfig:
        if self.width % 2 or self.height % 2:
            raise ValueError("export width and height must be even")
        if self.height <= self.width:
            raise ValueError("export must be vertical (height > width)")
        return self


class SubtitleConfig(_Section):
    model: str = "small"
    language: str | None = None
    compute_type: str = "int8"


class TagDefinition(_Section):
    """A concept the zero-shot tagger can recognise. ``prompt`` is what gets embedded."""

    id: str
    label: str
    prompt: str
    label_pt: str = ""


class TagConfig(_Section):
    max_tags_per_segment: int = Field(default=3, ge=0)
    # A tag must stand out within the video (z-score) *and* clear an absolute cosine floor.
    min_zscore: float = 1.0
    min_similarity: float = Field(default=0.65, ge=0, le=1)
    # Cosine band mapped to 0-1 for the "reaction presence" signal.
    reaction_floor: float = 0.62
    reaction_ceiling: float = 0.76
    concepts: tuple[TagDefinition, ...] = ()
    reaction_cues: tuple[str, ...] = ()


class Settings(_Section):
    data_dir: Path
    upload: UploadConfig = UploadConfig()
    video: VideoConfig = VideoConfig()
    scenes: SceneConfig = SceneConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    search: SearchConfig = SearchConfig()
    scoring: ScoreWeights = ScoreWeights()
    export: ExportConfig = ExportConfig()
    subtitles: SubtitleConfig = SubtitleConfig()
    tags: TagConfig = TagConfig()

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def previews_dir(self) -> Path:
        return self.data_dir / "previews"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clipping_machine.sqlite3"

    def ensure_dirs(self) -> None:
        for directory in (
            self.uploads_dir,
            self.processed_dir,
            self.index_dir,
            self.previews_dir,
            self.clips_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def load_settings(
    config_dir: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Build :class:`Settings` from the YAML files and environment overrides.

    Environment variables (all optional): ``GCM_CONFIG_DIR``, ``GCM_DATA_DIR``,
    ``GCM_EMBEDDING_MODEL`` (hub id or local folder), ``GCM_DEVICE``, ``GCM_OFFLINE``.
    """
    source = os.environ if env is None else env
    directory = config_dir or Path(source.get("GCM_CONFIG_DIR") or DEFAULT_CONFIG_DIR)

    merged: dict[str, Any] = {}
    for name in ("processing.yaml", "models.yaml", "tags.yaml"):
        merged.update(_read_yaml(directory / name))

    embedding = dict(merged.get("embedding") or {})
    if source.get("GCM_EMBEDDING_MODEL"):
        embedding["model"] = source["GCM_EMBEDDING_MODEL"]
    if source.get("GCM_DEVICE"):
        embedding["device"] = source["GCM_DEVICE"]
    if source.get("GCM_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        embedding["local_files_only"] = True
    if embedding:
        merged["embedding"] = embedding

    data_dir = Path(source.get("GCM_DATA_DIR") or merged.pop("data_dir", None) or "data")
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    merged.pop("data_dir", None)
    return Settings(data_dir=data_dir, **merged)
