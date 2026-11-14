"""Indexing pipeline: scenes → keyframes/motion → audio → embeddings → stores.

Everything runs on this machine. Intermediate audio is deleted when indexing ends;
keyframes are kept because the UI uses them as previews.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from clipping_machine import ffmpeg
from clipping_machine.analysis import audio_analyzer
from clipping_machine.analysis.normalization import robust_normalize
from clipping_machine.analysis.scene_describer import SceneDescriber
from clipping_machine.config import Settings
from clipping_machine.embeddings.base import EmbeddingBackend
from clipping_machine.embeddings.video_embedder import build_window_input, embed_in_batches
from clipping_machine.ingest import scene_splitter
from clipping_machine.ingest.audio_chunker import AudioTrack, extract_audio
from clipping_machine.ingest.frame_extractor import extract_frames
from clipping_machine.ingest.windows import compute_windows
from clipping_machine.search.vector_store import VectorRecord, VectorStore
from clipping_machine.storage.models import Scene, Segment
from clipping_machine.storage.repositories import (
    SceneRepository,
    SegmentRepository,
    VideoRepository,
)

StageCallback = Callable[[str, float], None]

# (stage name, share of the total progress bar)
STAGES: tuple[tuple[str, float], ...] = (
    ("metadata", 0.01),
    ("scenes", 0.14),
    ("frames", 0.20),
    ("audio", 0.05),
    ("embeddings", 0.56),
    ("storing", 0.04),
)


class IndexingError(RuntimeError):
    pass


def segment_id_for(video_id: str, index: int) -> str:
    return f"{video_id}_seg_{index:06d}"


class _Progress:
    """Turns per-stage progress into one overall fraction persisted in SQLite."""

    def __init__(
        self, video_id: str, videos: VideoRepository, listener: StageCallback | None
    ) -> None:
        self._video_id = video_id
        self._videos = videos
        self._listener = listener
        self._offsets: dict[str, tuple[float, float]] = {}
        offset = 0.0
        for name, weight in STAGES:
            self._offsets[name] = (offset, weight)
            offset += weight
        self._last_write = 0.0

    def update(self, stage: str, fraction: float) -> None:
        offset, weight = self._offsets[stage]
        overall = offset + weight * max(0.0, min(1.0, fraction))
        now = time.monotonic()
        # Throttle database writes; always record stage boundaries.
        if fraction in (0.0, 1.0) or now - self._last_write >= 0.5:
            self._videos.set_progress(self._video_id, stage, overall)
            self._last_write = now
        if self._listener is not None:
            self._listener(stage, overall)

    def stage(self, name: str) -> Callable[[float], None]:
        self.update(name, 0.0)
        return lambda fraction: self.update(name, fraction)


class Indexer:
    def __init__(
        self,
        settings: Settings,
        backend: EmbeddingBackend,
        store: VectorStore,
        videos: VideoRepository,
        scenes: SceneRepository,
        segments: SegmentRepository,
        describer: SceneDescriber | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._store = store
        self._videos = videos
        self._scenes = scenes
        self._segments = segments
        self._describer = describer or SceneDescriber(backend, settings.tags)

    def previews_dir(self, video_id: str) -> Path:
        return self._settings.previews_dir / video_id

    def work_dir(self, video_id: str) -> Path:
        return self._settings.processed_dir / video_id

    def index_video(self, video_id: str, on_progress: StageCallback | None = None) -> int:
        """Index (or re-index) one video. Returns the number of segments created."""
        video = self._videos.require(video_id)
        self._videos.mark_indexing(video_id)
        started = time.perf_counter()
        try:
            count = self._run(video_id, Path(video.path), on_progress)
        except Exception as exc:
            self._videos.mark_failed(video_id, f"{type(exc).__name__}: {exc}")
            raise
        finally:
            shutil.rmtree(self.work_dir(video_id), ignore_errors=True)
        self._videos.mark_indexed(
            video_id,
            segments_count=count,
            index_seconds=time.perf_counter() - started,
            embedding_model=self._backend.model_name,
        )
        return count

    def _run(self, video_id: str, path: Path, on_progress: StageCallback | None) -> int:
        settings = self._settings
        progress = _Progress(video_id, self._videos, on_progress)

        # 1. Metadata (re-read: the file may have changed since import).
        report = progress.stage("metadata")
        if not path.is_file():
            raise IndexingError(f"the video file is missing: {path}")
        info = ffmpeg.probe(path)
        if info.duration_ms <= 0:
            raise IndexingError("the video has no measurable duration")
        self._clear(video_id)
        report(1.0)

        # 2. Scenes.
        report = progress.stage("scenes")
        scenes: list[Scene] = []
        if settings.scenes.enabled:
            scenes = scene_splitter.detect_scenes(
                path, video_id, info.duration_ms, info.fps, settings.scenes
            )
        if not scenes:
            scenes = [Scene(video_id, 0, 0, info.duration_ms)]
        self._scenes.replace(video_id, scenes)
        cuts = scene_splitter.cut_times_ms(scenes)
        report(1.0)

        # 3. Keyframes + motion (one decode pass).
        report = progress.stage("frames")
        frames = extract_frames(
            path, self.previews_dir(video_id), info, settings.video, on_progress=report
        )
        report(1.0)

        # 4. Audio.
        report = progress.stage("audio")
        track: AudioTrack | None = None
        if info.has_audio:
            wav = extract_audio(
                path,
                self.work_dir(video_id) / "audio.wav",
                settings.video.audio_sample_rate,
                duration_ms=info.duration_ms,
                on_progress=report,
            )
            track = AudioTrack(wav)
        report(1.0)

        try:
            windows = compute_windows(
                info.duration_ms, settings.video.window_ms, settings.video.overlap_ms
            )
            if not windows:
                raise IndexingError("the video is too short to index")
            inputs = [build_window_input(window, frames, track) for window in windows]
            audio_stats = [
                audio_analyzer.measure(item.audio, item.sample_rate)
                if item.audio is not None
                else audio_analyzer.SILENT
                for item in inputs
            ]

            # 5. Embeddings.
            report = progress.stage("embeddings")
            vectors = embed_in_batches(
                self._backend, inputs, settings.embedding.batch_size, on_progress=report
            )
            report(1.0)
        finally:
            if track is not None:
                track.close()

        # 6. Derived signals and storage.
        report = progress.stage("storing")
        audio_energy = audio_analyzer.normalize_intensity(audio_stats)
        motion_raw = np.asarray(
            [frames.motion_between(window.start_ms, window.end_ms) for window in windows],
            dtype=np.float32,
        )
        motion = robust_normalize(motion_raw, min_range=0.25)
        descriptions = self._describer.describe(vectors)

        segments: list[Segment] = []
        records: list[VectorRecord] = []
        for position, window in enumerate(windows):
            segment_id = segment_id_for(video_id, window.index)
            preview = inputs[position].frames[len(inputs[position].frames) // 2]
            segments.append(
                Segment(
                    segment_id=segment_id,
                    video_id=video_id,
                    seg_index=window.index,
                    start_ms=window.start_ms,
                    end_ms=window.end_ms,
                    preview_frame_path=str(preview),
                    embedding_model=self._backend.model_name,
                    embedding_dimension=self._backend.dimension,
                    audio_energy=float(audio_energy[position]),
                    audio_db=audio_stats[position].intensity_db,
                    motion=float(motion[position]),
                    motion_raw=float(motion_raw[position]),
                    scene_cuts=scene_splitter.count_cuts(cuts, window.start_ms, window.end_ms),
                    reaction=descriptions[position].reaction,
                    tags=list(descriptions[position].tags),
                )
            )
            records.append(
                VectorRecord(
                    segment_id, video_id, window.start_ms, window.end_ms, vectors[position]
                )
            )

        self._store.add(records)
        self._segments.replace(video_id, segments)
        report(1.0)
        return len(segments)

    def _clear(self, video_id: str) -> None:
        """Remove anything a previous indexing run left behind."""
        self._store.delete_video(video_id)
        self._segments.delete_for_video(video_id)
        shutil.rmtree(self.previews_dir(video_id), ignore_errors=True)
        shutil.rmtree(self.work_dir(video_id), ignore_errors=True)
