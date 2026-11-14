"""Interval merging, the vector index, and query → moments."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from clipping_machine.config import CalibrationConfig, Settings
from clipping_machine.search.interval_merger import (
    Hit,
    MergedInterval,
    build_intervals,
    expand_to_minimum,
    merge_hits,
    split_long,
)
from clipping_machine.search.query_engine import QueryEngine, calibrate
from clipping_machine.search.vector_store import VectorRecord, VectorStore
from clipping_machine.services import ClippingService
from clipping_machine.storage.database import utc_now
from clipping_machine.storage.models import Segment, Video, VideoStatus
from tests.fakes import FakeEmbedder


def _hit(start_s: float, end_s: float, similarity: float = 0.8, name: str | None = None) -> Hit:
    return Hit(name or f"s{start_s}", int(start_s * 1000), int(end_s * 1000), similarity)


# ----------------------------------------------------------- interval merging


def test_overlapping_and_adjacent_hits_merge() -> None:
    merged = merge_hits([_hit(6, 14), _hit(0, 8), _hit(12, 20)], 3000)
    assert [(m.start_ms, m.end_ms) for m in merged] == [(0, 20_000)]
    assert len(merged[0].hits) == 3


def test_gap_under_three_seconds_merges_and_three_or_more_does_not() -> None:
    assert len(merge_hits([_hit(0, 8), _hit(10.9, 18)], 3000)) == 1
    apart = merge_hits([_hit(0, 8), _hit(11, 19)], 3000)
    assert [(m.start_ms, m.end_ms) for m in apart] == [(0, 8000), (11_000, 19_000)]


def test_merge_keeps_the_longest_end_and_handles_empty_input() -> None:
    merged = merge_hits([_hit(0, 30), _hit(5, 10)], 3000)
    assert (merged[0].start_ms, merged[0].end_ms) == (0, 30_000)
    assert merge_hits([], 3000) == []


def test_interval_similarity_is_peak_weighted() -> None:
    interval = merge_hits([_hit(0, 8, 0.9), _hit(6, 14, 0.7)], 3000)[0]
    assert interval.similarity == pytest.approx(0.7 * 0.9 + 0.3 * 0.8)
    assert interval.best_hit.similarity == 0.9


def test_short_interval_is_expanded_to_the_minimum() -> None:
    short = MergedInterval(20_000, 24_000, (_hit(20, 24),))
    grown = expand_to_minimum(short, 8000, 60_000)
    assert (grown.start_ms, grown.end_ms) == (18_000, 26_000)
    assert expand_to_minimum(grown, 8000, 60_000) is grown


def test_expansion_stays_inside_the_video() -> None:
    at_start = expand_to_minimum(MergedInterval(0, 2000, (_hit(0, 2),)), 8000, 60_000)
    assert (at_start.start_ms, at_start.end_ms) == (0, 8000)
    at_end = expand_to_minimum(MergedInterval(58_000, 60_000, (_hit(58, 60),)), 8000, 60_000)
    assert (at_end.start_ms, at_end.end_ms) == (52_000, 60_000)
    tiny_video = expand_to_minimum(MergedInterval(0, 2000, (_hit(0, 2),)), 8000, 5000)
    assert (tiny_video.start_ms, tiny_video.end_ms) == (0, 5000)


def test_long_interval_is_split_into_target_sized_parts() -> None:
    hits = tuple(_hit(start, start + 8) for start in range(0, 96, 6))
    long = merge_hits(hits, 3000)[0]
    assert long.duration_ms == 98_000
    parts = split_long(long, 60_000, 45_000)
    assert len(parts) == 3
    assert parts[0].start_ms == 0 and parts[-1].end_ms == 98_000
    assert all(part.duration_ms <= 45_000 for part in parts)
    assert all(a.end_ms == b.start_ms for a, b in itertools.pairwise(parts))
    assert all(part.hits for part in parts)


def test_interval_at_the_limit_is_not_split() -> None:
    interval = MergedInterval(0, 60_000, (_hit(0, 60),))
    assert split_long(interval, 60_000, 45_000) == [interval]


def test_build_intervals_applies_every_rule() -> None:
    hits = [_hit(0, 8), _hit(6, 14), _hit(40, 42), *(_hit(s, s + 8) for s in range(100, 190, 6))]
    intervals = build_intervals(
        hits,
        merge_gap_ms=3000,
        min_ms=8000,
        max_ms=60_000,
        target_max_ms=45_000,
        video_duration_ms=300_000,
    )
    spans = [(i.start_ms, i.end_ms) for i in intervals]
    assert spans[0] == (0, 14_000)
    assert spans[1] == (37_000, 45_000)  # 2 s hit expanded to 8 s
    assert all(8000 <= end - start <= 60_000 for start, end in spans)
    assert spans[-1][1] == 192_000


# --------------------------------------------------------------- vector store


def _unit(dimension: int, axis: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float32)
    vector[axis] = 1.0
    return vector


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "index", 16, "fake/model")


def test_store_search_orders_by_cosine_similarity(store: VectorStore) -> None:
    mixed = (_unit(16, 0) + _unit(16, 1)) / np.sqrt(2)
    store.add(
        [
            VectorRecord("a", "vid_1", 0, 8000, _unit(16, 0)),
            VectorRecord("b", "vid_1", 6000, 14_000, mixed.astype(np.float32)),
            VectorRecord("c", "vid_1", 12_000, 20_000, _unit(16, 2)),
        ]
    )
    hits = store.search(_unit(16, 0), top_k=3)
    assert [hit.segment_id for hit in hits] == ["a", "b", "c"]
    assert hits[0].similarity == pytest.approx(1.0, abs=1e-5)
    assert hits[1].similarity == pytest.approx(0.7071, abs=1e-3)
    assert hits[2].similarity == pytest.approx(0.0, abs=1e-5)
    assert (hits[1].start_ms, hits[1].end_ms) == (6000, 14_000)


def test_store_filters_by_video_and_respects_top_k(store: VectorStore) -> None:
    store.add(
        [
            VectorRecord("a", "vid_1", 0, 8000, _unit(16, 0)),
            VectorRecord("b", "vid_2", 0, 8000, _unit(16, 0)),
            VectorRecord("c", "vid_2", 6000, 14_000, _unit(16, 1)),
        ]
    )
    assert {hit.segment_id for hit in store.search(_unit(16, 0), top_k=10, video_id="vid_2")} == {
        "b",
        "c",
    }
    assert len(store.search(_unit(16, 0), top_k=1)) == 1
    assert store.count() == 3 and store.count("vid_2") == 2


def test_store_delete_video(store: VectorStore) -> None:
    store.add(
        [
            VectorRecord("a", "vid_1", 0, 8000, _unit(16, 0)),
            VectorRecord("b", "vid_2", 0, 8000, _unit(16, 1)),
        ]
    )
    store.delete_video("vid_1")
    assert store.count() == 1
    assert [hit.segment_id for hit in store.search(_unit(16, 0), top_k=5)] == ["b"]


def test_store_vectors_between(store: VectorStore) -> None:
    store.add(
        [
            VectorRecord("a", "vid_1", 0, 8000, _unit(16, 0)),
            VectorRecord("b", "vid_1", 6000, 14_000, _unit(16, 1)),
            VectorRecord("c", "vid_1", 12_000, 20_000, _unit(16, 2)),
        ]
    )
    assert store.vectors_between("vid_1", 7000, 12_000).shape == (2, 16)
    assert store.vectors_between("vid_1", 30_000, 40_000).shape == (0, 16)
    assert store.vectors_between("vid_9", 0, 8000).shape == (0, 16)


def test_store_persists_on_disk(tmp_path: Path) -> None:
    VectorStore(tmp_path / "index", 16, "fake/model").add(
        [VectorRecord("a", "vid_1", 0, 8000, _unit(16, 0))]
    )
    reopened = VectorStore(tmp_path / "index", 16, "fake/model")
    assert reopened.count() == 1


def test_store_never_mixes_embedding_models(tmp_path: Path) -> None:
    VectorStore(tmp_path / "index", 16, "model/one").add(
        [VectorRecord("a", "vid_1", 0, 8000, _unit(16, 0))]
    )
    other = VectorStore(tmp_path / "index", 16, "model/two")
    assert other.search(_unit(16, 0), top_k=5) == []


def test_store_rejects_bad_input(store: VectorStore) -> None:
    with pytest.raises(ValueError):
        store.add([VectorRecord("a", "vid_1", 0, 8000, np.zeros(8, dtype=np.float32))])
    nan = _unit(16, 0)
    nan[3] = np.nan
    with pytest.raises(ValueError):
        store.add([VectorRecord("a", "vid_1", 0, 8000, nan)])
    with pytest.raises(ValueError):
        store.search(np.zeros(8, dtype=np.float32), top_k=1)
    with pytest.raises(ValueError):
        store.search(_unit(16, 0), top_k=1, video_id="x' OR '1'='1")
    store.add([])
    assert store.count() == 0


# --------------------------------------------------------------- query engine


def test_calibrate_spreads_the_cosine_band() -> None:
    config = CalibrationConfig(floor=0.5, ceiling=0.8)
    assert calibrate(0.5, config) == 0.0
    assert calibrate(0.65, config) == pytest.approx(0.5)
    assert calibrate(0.8, config) == 1.0
    assert calibrate(0.3, config) == 0.0 and calibrate(0.95, config) == 1.0
    assert calibrate(0.65, CalibrationConfig(enabled=False)) == 0.65


def test_query_engine_rejects_empty_queries(tmp_path: Path) -> None:
    backend = FakeEmbedder(dimension=16)
    engine = QueryEngine(
        backend, VectorStore(tmp_path, 16, backend.model_name), CalibrationConfig()
    )
    with pytest.raises(ValueError):
        engine.search("   ", top_k=5)


# -------------------------------------------------------------- moment finder


def _seed(service: ClippingService, concepts: dict[int, str], windows: int = 30) -> Video:
    """Register an indexed video with one 8 s segment every 6 s, without FFmpeg."""
    backend = service.backend
    assert isinstance(backend, FakeEmbedder)
    backend.window_concepts = concepts
    backend.reset()
    now = utc_now()
    video = Video(
        video_id="vid_seed",
        original_name="seed.mp4",
        path="seed.mp4",
        managed_copy=False,
        size_bytes=1,
        fingerprint="seed",
        duration_ms=windows * 6000 + 2000,
        width=1920,
        height=1080,
        fps=60.0,
        video_codec="h264",
        audio_codec="aac",
        rights_confirmed_at=now,
        created_at=now,
        status=VideoStatus.INDEXED,
    )
    service.videos.add(video)
    from clipping_machine.embeddings.base import WindowInput

    vectors = backend.embed_windows([WindowInput(frames=(Path("f.jpg"),))] * windows)
    segments, records = [], []
    for index in range(windows):
        start, end = index * 6000, index * 6000 + 8000
        segment_id = f"vid_seed_seg_{index:06d}"
        segments.append(
            Segment(
                segment_id=segment_id,
                video_id=video.video_id,
                seg_index=index,
                start_ms=start,
                end_ms=end,
                preview_frame_path=f"kf_{index}.jpg",
                embedding_model=backend.model_name,
                embedding_dimension=backend.dimension,
                audio_energy=0.9 if index in (10, 11) else 0.1,
                motion=0.8 if index in (10, 11) else 0.2,
                tags=[concepts[index]] if index in concepts else [],
            )
        )
        records.append(VectorRecord(segment_id, video.video_id, start, end, vectors[index]))
    service.segments.replace(video.video_id, segments)
    service.store.add(records)
    return video


def test_search_returns_merged_intervals_not_isolated_segments(service: ClippingService) -> None:
    _seed(service, {10: "explosion", 11: "explosion", 20: "explosion", 3: "drift"})
    result = service.search("big explosion", fill_to_min_results=False)
    spans = sorted((m.start_ms, m.end_ms) for m in result.moments)
    assert spans == [(60_000, 74_000), (120_000, 128_000)]
    assert result.hits_above_threshold == 3
    assert all(not m.below_threshold for m in result.moments)
    assert all(m.similarity > 0.9 for m in result.moments)


def test_results_are_ranked_by_editorial_score(service: ClippingService) -> None:
    _seed(service, {10: "explosion", 11: "explosion", 20: "explosion"})
    result = service.search("explosion", fill_to_min_results=False)
    first, second = result.moments
    # Same semantic match; the first one is louder, has more motion and a better length.
    assert (first.start_ms, first.end_ms) == (60_000, 74_000)
    assert first.score > second.score
    assert first.breakdown.signals.audio_intensity > second.breakdown.signals.audio_intensity
    assert "loud_audio" in first.highlights and "high_motion" in first.highlights
    assert first.tags == ("explosion",)
    assert first.preview_frame_path in {"kf_10.jpg", "kf_11.jpg"}
    assert 0.0 <= second.score <= first.score <= 1.0


def test_min_similarity_filters_unrelated_segments(service: ClippingService) -> None:
    _seed(service, {10: "explosion"})
    strict = service.search("drift", fill_to_min_results=False)
    assert strict.moments == ()
    assert strict.hits_above_threshold == 0
    assert strict.hits_total > 0


def test_fill_to_min_results_marks_weak_moments(service: ClippingService) -> None:
    _seed(service, {10: "explosion"})
    result = service.search("explosion")
    strong = [m for m in result.moments if not m.below_threshold]
    weak = [m for m in result.moments if m.below_threshold]
    assert len(strong) == 1 and (strong[0].start_ms, strong[0].end_ms) == (60_000, 68_000)
    assert result.moments[0] is strong[0]
    assert weak and result.filled == len(weak)
    assert len(result.moments) <= service.settings.search.min_results
    assert all(m.end_ms <= strong[0].start_ms or m.start_ms >= strong[0].end_ms for m in weak)


def test_search_can_return_at_least_ten_candidates(service: ClippingService) -> None:
    _seed(service, dict.fromkeys(range(0, 60, 3), "chase"), windows=60)
    result = service.search("police chase", fill_to_min_results=False)
    assert len(result.moments) >= 10


def test_search_scope_and_logging(service: ClippingService) -> None:
    _seed(service, {10: "explosion"})
    assert service.search("explosion", video_id="vid_other").moments == ()
    assert service.search("explosion", video_id="vid_seed").moments
    count, latency = service.search_log.stats()
    assert count == 2 and latency >= 0


def test_novelty_penalises_moments_already_clipped(service: ClippingService) -> None:
    _seed(service, {10: "explosion", 20: "explosion"})
    before = service.search("explosion", fill_to_min_results=False)
    target = next(m for m in before.moments if m.start_ms == 60_000)
    assert target.breakdown.signals.novelty == 1.0
    service.create_clip(target, query="explosion", reason="test")

    after = service.search("explosion", fill_to_min_results=False)
    again = next(m for m in after.moments if m.start_ms == 60_000)
    assert again.breakdown.signals.novelty == 0.0
    assert again.score < target.score
    assert "similar_to_existing_clip" in again.highlights


def test_rescore_after_manual_trim(service: ClippingService) -> None:
    _seed(service, {10: "explosion"})
    moment = service.search("explosion", fill_to_min_results=False).moments[0]
    assert moment.duration_ms == 8000
    longer = service.finder.rescore(moment, 56_000, 76_000)
    assert (longer.start_ms, longer.end_ms) == (56_000, 76_000)
    assert longer.breakdown.signals.duration_fit == 1.0
    assert longer.breakdown.signals.duration_fit > moment.breakdown.signals.duration_fit
    assert longer.similarity == moment.similarity


def test_settings_are_injected(settings: Settings, service: ClippingService) -> None:
    assert service.settings is settings
    assert service.store.dimension == 768
