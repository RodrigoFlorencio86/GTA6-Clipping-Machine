"""From a text query to ranked, clip-sized moments with an editorial score."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from clipping_machine.analysis.highlight_scorer import (
    ScoreBreakdown,
    ScoreSignals,
    compute_score,
    duration_fit,
    novelty,
)
from clipping_machine.config import Settings
from clipping_machine.embeddings.base import Vector
from clipping_machine.publishing.approval_queue import ACTIVE_STATUSES
from clipping_machine.search.interval_merger import Hit, MergedInterval, build_intervals
from clipping_machine.search.query_engine import QueryEngine
from clipping_machine.search.vector_store import VectorStore
from clipping_machine.storage.models import Segment
from clipping_machine.storage.repositories import (
    ClipRepository,
    SearchLogRepository,
    SegmentRepository,
    VideoRepository,
)

# Signal levels that earn a mention in the human-readable reason.
_NOTABLE = 0.6


@dataclass(frozen=True)
class Moment:
    video_id: str
    start_ms: int
    end_ms: int
    score: float
    breakdown: ScoreBreakdown
    similarity: float
    tags: tuple[str, ...]
    highlights: tuple[str, ...]
    preview_frame_path: str
    segment_ids: tuple[str, ...]
    below_threshold: bool = False

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.start_ms}-{self.end_ms}"


@dataclass(frozen=True)
class SearchResult:
    query: str
    moments: tuple[Moment, ...]
    hits_total: int
    hits_above_threshold: int
    elapsed_ms: float
    min_similarity: float
    filled: int = 0
    best_similarity: float = field(default=0.0)
    # The cut actually applied: ``max(min_similarity, best_similarity - relative_margin)``.
    effective_threshold: float = field(default=0.0)


def effective_threshold(min_similarity: float, best_similarity: float, margin: float) -> float:
    """Absolute floor, raised to ``best - margin`` when the query has a clear best match.

    EmbeddingGemma 2 similarities sit in a narrow band, so on some footage most windows
    clear the absolute floor. Keeping only those close to the best match stops loosely
    related but loud or busy windows from outranking the real match, and keeps merged
    moments tight. Above-threshold windows that miss the cut become fill-in candidates.
    """
    if margin <= 0:
        return min_similarity
    return max(min_similarity, best_similarity - margin)


@dataclass
class _PriorClips:
    """Clips already made: the reference for the novelty signal."""

    intervals_by_video: dict[str, list[tuple[int, int]]]
    vectors: list[Vector]


def _mean_vector(vectors: np.ndarray) -> Vector | None:
    if vectors.shape[0] == 0:
        return None
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return (mean / norm).astype(np.float32) if norm > 0 else None


def _blend(values: Sequence[float]) -> float:
    """Peak-weighted aggregate: a moment is as strong as its best part, mostly."""
    if not values:
        return 0.0
    return 0.5 * max(values) + 0.5 * (sum(values) / len(values))


def _overlaps(a: MergedInterval, b: MergedInterval) -> bool:
    return a.start_ms < b.end_ms and b.start_ms < a.end_ms


class MomentFinder:
    def __init__(
        self,
        settings: Settings,
        engine: QueryEngine,
        store: VectorStore,
        videos: VideoRepository,
        segments: SegmentRepository,
        clips: ClipRepository,
        search_log: SearchLogRepository | None = None,
    ) -> None:
        self._settings = settings
        self._engine = engine
        self._store = store
        self._videos = videos
        self._segments = segments
        self._clips = clips
        self._search_log = search_log

    # ------------------------------------------------------------------ search

    def find(
        self,
        query: str,
        *,
        video_id: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        relative_margin: float | None = None,
        merge_gap_seconds: float | None = None,
        fill_to_min_results: bool = True,
    ) -> SearchResult:
        """Search, merge neighbouring hits into intervals, score and rank them."""
        config = self._settings.search
        floor = config.min_similarity if min_similarity is None else min_similarity
        margin = config.relative_margin if relative_margin is None else relative_margin
        gap_ms = config.merge_gap_ms if merge_gap_seconds is None else int(merge_gap_seconds * 1000)
        self._engine.warm_up()
        started = time.perf_counter()

        vector_hits = self._engine.search(query, top_k=top_k or config.top_k, video_id=video_id)
        best_similarity = max((hit.similarity for hit in vector_hits), default=0.0)
        threshold = effective_threshold(floor, best_similarity, margin)
        by_video: dict[str, list[Hit]] = defaultdict(list)
        for vector_hit in vector_hits:
            by_video[vector_hit.video_id].append(self._engine.to_hit(vector_hit))

        prior = self._prior_clips()
        strong: list[Moment] = []
        weak: list[Moment] = []
        above = 0
        for vid, hits in by_video.items():
            video = self._videos.get(vid)
            if video is None:
                continue
            passing = [hit for hit in hits if hit.similarity >= threshold]
            failing = [hit for hit in hits if hit.similarity < threshold]
            above += len(passing)
            strong_intervals = self._intervals(passing, gap_ms, video.duration_ms)
            strong += [self._moment(vid, video.duration_ms, i, prior) for i in strong_intervals]
            if fill_to_min_results and failing:
                for interval in self._intervals(failing, gap_ms, video.duration_ms):
                    if any(_overlaps(interval, other) for other in strong_intervals):
                        continue
                    weak.append(
                        self._moment(vid, video.duration_ms, interval, prior, below_threshold=True)
                    )

        strong.sort(key=lambda moment: moment.score, reverse=True)
        weak.sort(key=lambda moment: moment.similarity, reverse=True)
        missing = max(0, config.min_results - len(strong)) if fill_to_min_results else 0
        filler = weak[:missing]
        elapsed_ms = (time.perf_counter() - started) * 1000

        moments = (*strong, *filler)
        if self._search_log is not None:
            self._search_log.add(video_id, query.strip(), len(moments), elapsed_ms)
        return SearchResult(
            query=query.strip(),
            moments=moments,
            hits_total=len(vector_hits),
            hits_above_threshold=above,
            elapsed_ms=elapsed_ms,
            min_similarity=floor,
            filled=len(filler),
            best_similarity=best_similarity,
            effective_threshold=threshold,
        )

    # --------------------------------------------------------------- rescoring

    def rescore(
        self, moment: Moment, start_ms: int, end_ms: int, *, exclude_clip_id: str | None = None
    ) -> Moment:
        """Recompute the score after the user adjusted the start/end by hand."""
        video = self._videos.require(moment.video_id)
        interval = MergedInterval(
            start_ms,
            end_ms,
            (Hit("manual", start_ms, end_ms, moment.similarity),),
        )
        rescored = self._moment(
            moment.video_id,
            video.duration_ms,
            interval,
            self._prior_clips(exclude_clip_id),
            below_threshold=moment.below_threshold,
        )
        return Moment(
            video_id=rescored.video_id,
            start_ms=rescored.start_ms,
            end_ms=rescored.end_ms,
            score=rescored.score,
            breakdown=rescored.breakdown,
            similarity=moment.similarity,
            tags=rescored.tags or moment.tags,
            highlights=rescored.highlights,
            preview_frame_path=rescored.preview_frame_path or moment.preview_frame_path,
            segment_ids=rescored.segment_ids,
            below_threshold=moment.below_threshold,
        )

    # ---------------------------------------------------------------- internals

    def _intervals(
        self, hits: Sequence[Hit], gap_ms: int, video_duration_ms: int
    ) -> list[MergedInterval]:
        config = self._settings.search
        return build_intervals(
            hits,
            merge_gap_ms=gap_ms,
            min_ms=config.min_clip_ms,
            max_ms=config.max_clip_ms,
            target_max_ms=config.target_ms[1],
            video_duration_ms=video_duration_ms,
        )

    def _prior_clips(self, exclude_clip_id: str | None = None) -> _PriorClips:
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        vectors: list[Vector] = []
        for clip in self._clips.find(statuses=ACTIVE_STATUSES):
            if clip.clip_id == exclude_clip_id:
                continue
            intervals[clip.video_id].append((clip.start_ms, clip.end_ms))
            mean = _mean_vector(
                self._store.vectors_between(clip.video_id, clip.start_ms, clip.end_ms)
            )
            if mean is not None:
                vectors.append(mean)
        return _PriorClips(dict(intervals), vectors)

    def _moment(
        self,
        video_id: str,
        video_duration_ms: int,
        interval: MergedInterval,
        prior: _PriorClips,
        *,
        below_threshold: bool = False,
    ) -> Moment:
        config = self._settings.search
        start_ms = max(0, interval.start_ms)
        end_ms = min(video_duration_ms, interval.end_ms)
        segments = self._segments.overlapping(video_id, start_ms, end_ms)
        vector = _mean_vector(self._store.vectors_between(video_id, start_ms, end_ms))

        signals = ScoreSignals(
            semantic_similarity=self._engine.score_component(interval.similarity),
            audio_intensity=_blend([segment.audio_energy for segment in segments]),
            visual_motion=_blend([segment.motion for segment in segments]),
            reaction_presence=max((segment.reaction for segment in segments), default=0.0),
            novelty=novelty(
                start_ms, end_ms, vector, prior.intervals_by_video.get(video_id, []), prior.vectors
            ),
            duration_fit=duration_fit(
                end_ms - start_ms, config.target_ms, config.min_clip_ms, config.max_clip_ms
            ),
        )
        breakdown = compute_score(signals, self._settings.scoring)
        best = interval.best_hit
        preview = next(
            (s.preview_frame_path for s in segments if s.segment_id == best.segment_id),
            segments[len(segments) // 2].preview_frame_path if segments else "",
        )
        return Moment(
            video_id=video_id,
            start_ms=start_ms,
            end_ms=end_ms,
            score=breakdown.total,
            breakdown=breakdown,
            similarity=interval.similarity,
            tags=_top_tags(segments, self._settings.tags.max_tags_per_segment),
            highlights=_highlights(breakdown.signals, segments),
            preview_frame_path=preview,
            segment_ids=tuple(segment.segment_id for segment in segments),
            below_threshold=below_threshold,
        )


def _top_tags(segments: Sequence[Segment], limit: int) -> tuple[str, ...]:
    weights: dict[str, float] = {}
    for segment in segments:
        for rank, tag in enumerate(segment.tags):
            weights[tag] = weights.get(tag, 0.0) + 1.0 / (rank + 1)
    ranked = sorted(weights, key=lambda tag: weights[tag], reverse=True)
    return tuple(ranked[: max(limit, 0)])


def _highlights(signals: ScoreSignals, segments: Sequence[Segment]) -> tuple[str, ...]:
    """Codes for the signals worth mentioning; the UI turns them into words."""
    codes: list[str] = []
    if signals.semantic_similarity >= _NOTABLE:
        codes.append("semantic_match")
    if signals.audio_intensity >= _NOTABLE:
        codes.append("loud_audio")
    if signals.visual_motion >= _NOTABLE:
        codes.append("high_motion")
    if signals.reaction_presence >= _NOTABLE:
        codes.append("reaction")
    if any(segment.scene_cuts for segment in segments):
        codes.append("scene_cut")
    if signals.novelty < 0.5:
        codes.append("similar_to_existing_clip")
    return tuple(codes)
