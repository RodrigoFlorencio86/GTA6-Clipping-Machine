"""Application wiring: one object that owns settings, stores and use-cases.

The Streamlit UI and the CLI both talk to :class:`ClippingService`; neither touches
FFmpeg, SQLite or the vector index directly.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import BinaryIO

from clipping_machine import ffmpeg, privacy
from clipping_machine.analysis.scene_describer import SceneDescriber
from clipping_machine.analysis.transcriber import Transcriber
from clipping_machine.clipping import subtitle_burner, thumbnail_generator, vertical_formatter
from clipping_machine.clipping.clip_builder import (
    RenderOptions,
    clip_filename,
    render_clip,
    unique_path,
)
from clipping_machine.clipping.vertical_formatter import Framing
from clipping_machine.config import Settings, load_settings
from clipping_machine.embeddings.base import EmbeddingBackend
from clipping_machine.embeddings.embeddinggemma_client import EmbeddingGemmaClient
from clipping_machine.ingest.audio_chunker import extract_audio
from clipping_machine.ingest.video_importer import VideoImporter
from clipping_machine.pipeline.indexer import Indexer
from clipping_machine.pipeline.jobs import JobManager
from clipping_machine.publishing.approval_queue import ApprovalQueue
from clipping_machine.search.moment_finder import Moment, MomentFinder, SearchResult
from clipping_machine.search.query_engine import QueryEngine
from clipping_machine.search.vector_store import VectorStore
from clipping_machine.storage.database import Database, utc_now
from clipping_machine.storage.models import Clip, ClipStatus, Video, VideoStatus
from clipping_machine.storage.repositories import (
    ClipRepository,
    SceneRepository,
    SearchLogRepository,
    SegmentRepository,
    VideoRepository,
)


class ClipTooLongError(ValueError):
    pass


@dataclass(frozen=True)
class ExportResult:
    clip: Clip
    output_path: Path
    thumbnail_path: Path | None
    subtitles_burned: bool


class ClippingService:
    def __init__(self, settings: Settings, backend: EmbeddingBackend | None = None) -> None:
        privacy.enforce_local_only()
        settings.ensure_dirs()
        self.settings = settings
        self.db = Database(settings.db_path)
        self.videos = VideoRepository(self.db)
        self.scenes = SceneRepository(self.db)
        self.segments = SegmentRepository(self.db)
        self.clips = ClipRepository(self.db)
        self.search_log = SearchLogRepository(self.db)

        self.backend: EmbeddingBackend = backend or EmbeddingGemmaClient(settings.embedding)
        self.store = VectorStore(
            settings.index_dir, self.backend.dimension, self.backend.model_name
        )
        self.importer = VideoImporter(settings, self.videos)
        self.indexer = Indexer(
            settings,
            self.backend,
            self.store,
            self.videos,
            self.scenes,
            self.segments,
            SceneDescriber(self.backend, settings.tags),
        )
        self.finder = MomentFinder(
            settings,
            QueryEngine(self.backend, self.store, settings.search.calibration),
            self.store,
            self.videos,
            self.segments,
            self.clips,
            self.search_log,
        )
        self.queue = ApprovalQueue(self.clips)
        self.jobs = JobManager(self.indexer.index_video)
        self._transcriber: Transcriber | None = None
        # A previous process may have died mid-indexing.
        self.videos.fail_interrupted()

    @classmethod
    def from_environment(cls) -> ClippingService:
        return cls(load_settings())

    # ------------------------------------------------------------------ videos

    def import_upload(self, stream: BinaryIO, name: str, *, rights_confirmed: bool) -> Video:
        return self.importer.import_upload(stream, name, rights_confirmed=rights_confirmed)

    def import_path(self, path: str, *, rights_confirmed: bool, copy: bool = False) -> Video:
        return self.importer.import_path(path, rights_confirmed=rights_confirmed, copy=copy)

    def start_indexing(self, video_id: str) -> bool:
        """Queue a video for background indexing."""
        self.videos.require(video_id)
        if self.jobs.is_active(video_id):
            return False
        # Mark first so the UI shows the queued state before the worker picks it up.
        self.videos.mark_indexing(video_id)
        return self.jobs.submit(video_id)

    def index_now(self, video_id: str) -> int:
        """Index synchronously (CLI, tests)."""
        return self.indexer.index_video(video_id)

    def delete_video(self, video_id: str) -> None:
        """Erase a video and everything derived from it: index, previews, clips, rows.

        The original file is deleted only if the app made its own copy of it.
        """
        video = self.videos.require(video_id)
        if self.jobs.is_active(video_id):
            raise RuntimeError("the video is being indexed; wait for it to finish")
        for clip in self.clips.find(video_id=video_id):
            self._remove_clip_files(clip)
        self.store.delete_video(video_id)
        shutil.rmtree(self.settings.previews_dir / video_id, ignore_errors=True)
        shutil.rmtree(self.settings.processed_dir / video_id, ignore_errors=True)
        shutil.rmtree(self.settings.clips_dir / video_id, ignore_errors=True)
        if video.managed_copy:
            Path(video.path).unlink(missing_ok=True)
        self.videos.delete(video_id)

    # ------------------------------------------------------------------ search

    def search(
        self,
        query: str,
        *,
        video_id: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        merge_gap_seconds: float | None = None,
        fill_to_min_results: bool = True,
    ) -> SearchResult:
        return self.finder.find(
            query,
            video_id=video_id,
            top_k=top_k,
            min_similarity=min_similarity,
            merge_gap_seconds=merge_gap_seconds,
            fill_to_min_results=fill_to_min_results,
        )

    def has_indexed_videos(self) -> bool:
        return bool(self.videos.all(VideoStatus.INDEXED))

    # ------------------------------------------------------------------- clips

    def validate_interval(self, video: Video, start_ms: int, end_ms: int) -> None:
        if start_ms < 0 or end_ms > video.duration_ms or end_ms <= start_ms:
            raise ValueError("the clip interval is outside the video")
        if end_ms - start_ms > self.settings.search.max_clip_ms:
            raise ClipTooLongError(
                f"clips are limited to {self.settings.search.max_clip_duration_seconds:g} s"
            )

    def create_clip(
        self,
        moment: Moment,
        *,
        query: str,
        reason: str,
        title: str = "",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> Clip:
        """Register a candidate clip (optionally with a manually adjusted interval)."""
        video = self.videos.require(moment.video_id)
        start = moment.start_ms if start_ms is None else start_ms
        end = moment.end_ms if end_ms is None else end_ms
        self.validate_interval(video, start, end)
        if (start, end) != (moment.start_ms, moment.end_ms):
            moment = self.finder.rescore(moment, start, end)
        now = utc_now()
        clip = Clip(
            clip_id=f"clip_{uuid.uuid4().hex[:12]}",
            video_id=video.video_id,
            start_ms=start,
            end_ms=end,
            query=query,
            title=title.strip(),
            score=moment.score,
            score_breakdown=moment.breakdown.as_dict(),
            reason=reason,
            created_at=now,
            updated_at=now,
            status=ClipStatus.CANDIDATE,
        )
        self.clips.add(clip)
        return clip

    def adjust_clip(self, clip_id: str, start_ms: int, end_ms: int) -> Clip:
        """Change a clip's interval while it is still under review."""
        clip = self.clips.require(clip_id)
        if clip.status not in (ClipStatus.CANDIDATE, ClipStatus.APPROVED):
            raise ValueError("only candidate or approved clips can be adjusted")
        self.validate_interval(self.videos.require(clip.video_id), start_ms, end_ms)
        if clip.draft_path:
            Path(clip.draft_path).unlink(missing_ok=True)
        self.clips.update_interval(clip_id, start_ms, end_ms)
        return self.clips.require(clip_id)

    def _clip_dir(self, clip: Clip) -> Path:
        return self.settings.clips_dir / clip.video_id

    def _focus(
        self, video: Video, info: ffmpeg.MediaInfo, clip: Clip, framing: Framing
    ) -> float | None:
        if framing != "smart":
            return None
        return vertical_formatter.estimate_focus_x(
            Path(video.path), info, clip.start_ms, clip.end_ms
        )

    def render_draft(
        self,
        clip_id: str,
        *,
        framing: Framing | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> Path:
        """Small, fast vertical preview for the reviewer. Not the final export."""
        clip = self.clips.require(clip_id)
        video = self.videos.require(clip.video_id)
        info = ffmpeg.probe(Path(video.path))
        chosen: Framing = framing or self.settings.export.framing
        options = RenderOptions.draft(
            self.settings.export, framing=chosen, focus_x=self._focus(video, info, clip, chosen)
        )
        output = self._clip_dir(clip) / "drafts" / f"{clip.clip_id}.mp4"
        render_clip(
            Path(video.path),
            info,
            clip.start_ms,
            clip.end_ms,
            output,
            options,
            on_progress=on_progress,
        )
        self.clips.set_draft(clip_id, str(output))
        return output

    def render_preview(self, video_id: str, start_ms: int, end_ms: int) -> Path:
        """Low-resolution preview of an arbitrary range (search results, before a clip exists)."""
        video = self.videos.require(video_id)
        info = ffmpeg.probe(Path(video.path))
        end_ms = min(end_ms, info.duration_ms)
        output = self.settings.previews_dir / video_id / "moments" / f"{start_ms}_{end_ms}.mp4"
        if output.is_file():
            return output
        options = RenderOptions.draft(self.settings.export, framing="center")
        return render_clip(Path(video.path), info, start_ms, end_ms, output, options)

    def export_clip(
        self,
        clip_id: str,
        *,
        framing: Framing | None = None,
        burn_subtitles: bool = False,
        on_progress: Callable[[float], None] | None = None,
    ) -> ExportResult:
        """Final 9:16 H.264 MP4. Refuses clips a human has not approved."""
        clip = self.queue.require_approved(clip_id)
        video = self.videos.require(clip.video_id)
        source = Path(video.path)
        info = ffmpeg.probe(source)
        chosen: Framing = framing or self.settings.export.framing
        focus = self._focus(video, info, clip, chosen)
        options = RenderOptions.final(self.settings.export, framing=chosen, focus_x=focus)

        directory = self._clip_dir(clip)
        output = unique_path(
            directory
            / clip_filename(
                self.settings.export.filename_prefix, date.today(), clip.start_ms, clip.score
            )
        )
        work = self.settings.processed_dir / clip.clip_id
        subtitles: Path | None = None
        try:
            if burn_subtitles:
                subtitles = self._make_subtitles(source, clip, work)
            render_clip(
                source,
                info,
                clip.start_ms,
                clip.end_ms,
                output,
                options,
                subtitles_path=subtitles,
                on_progress=on_progress,
            )
        finally:
            shutil.rmtree(work, ignore_errors=True)

        thumbnail: Path | None = output.with_suffix(".jpg")
        try:
            thumbnail_generator.generate_thumbnail(
                source,
                info,
                (clip.start_ms + clip.end_ms) // 2,
                output.with_suffix(".jpg"),
                width=options.width,
                height=options.height,
                framing=chosen,
                focus_x=focus,
            )
        except ffmpeg.FFmpegError:
            thumbnail = None

        self.clips.set_export(clip_id, str(output), str(thumbnail) if thumbnail else None)
        exported = self.queue.mark_exported(clip_id)
        return ExportResult(exported, output, thumbnail, subtitles is not None)

    def _make_subtitles(self, source: Path, clip: Clip, work: Path) -> Path | None:
        if self._transcriber is None:
            self._transcriber = Transcriber(
                self.settings.subtitles,
                local_files_only=self.settings.embedding.local_files_only,
            )
        wav = extract_audio(
            source,
            work / "clip.wav",
            16000,
            start_ms=clip.start_ms,
            clip_ms=clip.duration_ms,
        )
        return subtitle_burner.write_srt(self._transcriber.transcribe(wav), work)

    def delete_clip(self, clip_id: str) -> None:
        clip = self.clips.require(clip_id)
        self._remove_clip_files(clip)
        self.clips.delete(clip_id)

    @staticmethod
    def _remove_clip_files(clip: Clip) -> None:
        for candidate in (clip.draft_path, clip.output_path, clip.thumbnail_path):
            if candidate:
                Path(candidate).unlink(missing_ok=True)
