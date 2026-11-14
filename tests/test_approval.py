"""Approval queue and the human-in-the-loop export rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from clipping_machine import ffmpeg
from clipping_machine.publishing.approval_queue import (
    ALLOWED_TRANSITIONS,
    InvalidTransitionError,
    can_transition,
)
from clipping_machine.services import ClippingService, ClipTooLongError
from clipping_machine.storage.models import Clip, ClipStatus
from tests.fakes import FakeEmbedder


def _clip(service: ClippingService, embedder: FakeEmbedder, video_path: Path) -> Clip:
    embedder.window_concepts = {1: "chase"}
    embedder.reset()
    video = service.import_path(str(video_path), rights_confirmed=True)
    service.index_now(video.video_id)
    moment = service.search("chase", fill_to_min_results=False).moments[0]
    return service.create_clip(moment, query="chase", reason="police chase · loud audio")


def test_every_status_has_transition_rules() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(ClipStatus)
    assert ALLOWED_TRANSITIONS[ClipStatus.ARCHIVED] == frozenset()


def test_nothing_reaches_exported_or_published_without_approval() -> None:
    for status in (ClipStatus.CANDIDATE, ClipStatus.REJECTED):
        assert not can_transition(status, ClipStatus.EXPORTED)
        assert not can_transition(status, ClipStatus.PUBLISHED)
    assert not can_transition(ClipStatus.APPROVED, ClipStatus.PUBLISHED)
    assert can_transition(ClipStatus.APPROVED, ClipStatus.EXPORTED)
    assert can_transition(ClipStatus.EXPORTED, ClipStatus.PUBLISHED)


@pytest.mark.ffmpeg
def test_new_clips_start_as_candidates(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    clip = _clip(service, embedder, sample_video)
    assert clip.status is ClipStatus.CANDIDATE
    assert clip.query == "chase" and clip.reason == "police chase · loud audio"
    assert (clip.start_ms, clip.end_ms) == (6000, 14_000)
    assert set(clip.score_breakdown) == {
        "semantic_similarity",
        "audio_intensity",
        "visual_motion",
        "reaction_presence",
        "novelty",
        "duration_fit",
    }
    assert 0 < clip.score <= 1
    assert clip.output_path is None
    events = service.clips.events(clip.clip_id)
    assert [(e.from_status, e.to_status) for e in events] == [(None, ClipStatus.CANDIDATE)]


@pytest.mark.ffmpeg
def test_export_is_refused_until_a_human_approves(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    clip = _clip(service, embedder, sample_video)
    with pytest.raises(InvalidTransitionError):
        service.export_clip(clip.clip_id)
    assert service.clips.require(clip.clip_id).status is ClipStatus.CANDIDATE
    assert list(service.settings.clips_dir.rglob("*.mp4")) == []

    service.queue.reject(clip.clip_id, "not interesting")
    with pytest.raises(InvalidTransitionError):
        service.export_clip(clip.clip_id)


@pytest.mark.ffmpeg
def test_full_review_flow(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    clip = _clip(service, embedder, sample_video)
    draft = service.render_draft(clip.clip_id)
    assert ffmpeg.probe(draft).height == 640
    assert service.clips.require(clip.clip_id).draft_path == str(draft)

    service.queue.approve(clip.clip_id, "great chase")
    progress: list[float] = []
    result = service.export_clip(clip.clip_id, framing="center", on_progress=progress.append)

    exported = ffmpeg.probe(result.output_path)
    assert (exported.width, exported.height) == (1080, 1920)
    assert exported.video_codec == "h264"
    assert abs(exported.duration_ms - 8000) < 300
    assert result.output_path.parent == service.settings.clips_dir / clip.video_id
    assert result.output_path.name.startswith("gta6_clip_")
    assert "_00-00-06_score" in result.output_path.name
    assert result.thumbnail_path is not None and result.thumbnail_path.is_file()
    assert result.subtitles_burned is False
    assert progress[-1] == 1.0

    stored = service.clips.require(clip.clip_id)
    assert stored.status is ClipStatus.EXPORTED
    assert stored.output_path == str(result.output_path) and stored.exported_at

    # "Published" is only ever a manual bookkeeping step.
    service.queue.mark_published_manually(clip.clip_id)
    service.queue.archive(clip.clip_id)
    history = [event.to_status for event in service.clips.events(clip.clip_id)]
    assert history == [
        ClipStatus.CANDIDATE,
        ClipStatus.APPROVED,
        ClipStatus.EXPORTED,
        ClipStatus.PUBLISHED,
        ClipStatus.ARCHIVED,
    ]
    with pytest.raises(InvalidTransitionError):
        service.queue.reopen(clip.clip_id)


@pytest.mark.ffmpeg
def test_exporting_twice_keeps_both_files(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    clip = _clip(service, embedder, sample_video)
    service.queue.approve(clip.clip_id)
    first = service.export_clip(clip.clip_id, framing="center").output_path
    service.queue.transition(clip.clip_id, ClipStatus.APPROVED, "re-export")
    second = service.export_clip(clip.clip_id, framing="blur_pad").output_path
    assert first != second and first.is_file() and second.is_file()


@pytest.mark.ffmpeg
def test_manual_trim_and_limits(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    clip = _clip(service, embedder, sample_video)
    service.render_draft(clip.clip_id)
    adjusted = service.adjust_clip(clip.clip_id, 2000, 17_000)
    assert (adjusted.start_ms, adjusted.end_ms) == (2000, 17_000)
    assert adjusted.draft_path is None  # the old preview no longer matches

    with pytest.raises(ValueError):
        service.adjust_clip(clip.clip_id, 15_000, 99_000)  # past the end of the video
    with pytest.raises(ValueError):
        service.adjust_clip(clip.clip_id, 9000, 9000)

    service.queue.approve(clip.clip_id)
    service.export_clip(clip.clip_id, framing="center")
    with pytest.raises(ValueError):
        service.adjust_clip(clip.clip_id, 0, 8000)  # exported clips are frozen


@pytest.mark.ffmpeg
def test_clips_longer_than_sixty_seconds_are_rejected(
    service: ClippingService, embedder: FakeEmbedder, tmp_path: Path
) -> None:
    from tests.conftest import make_video

    long_video = make_video(tmp_path / "long.mp4", seconds=70, size="320x180", fps=5, audio=False)
    embedder.window_concepts = {1: "chase"}
    video = service.import_path(str(long_video), rights_confirmed=True)
    service.index_now(video.video_id)
    moment = service.search("chase", fill_to_min_results=False).moments[0]
    with pytest.raises(ClipTooLongError):
        service.create_clip(moment, query="chase", reason="r", start_ms=0, end_ms=61_000)
    clip = service.create_clip(moment, query="chase", reason="r", start_ms=0, end_ms=60_000)
    assert clip.duration_ms == 60_000
    # A hand-trimmed clip is rescored for its new length.
    assert clip.score_breakdown["duration_fit"] == 0.0


@pytest.mark.ffmpeg
def test_delete_clip_removes_its_files(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    clip = _clip(service, embedder, sample_video)
    draft = service.render_draft(clip.clip_id)
    service.delete_clip(clip.clip_id)
    assert not draft.exists()
    assert service.clips.get(clip.clip_id) is None


@pytest.mark.ffmpeg
def test_preview_of_a_search_result_is_cached(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    clip = _clip(service, embedder, sample_video)
    first = service.render_preview(clip.video_id, 6000, 14_000)
    stamp = first.stat().st_mtime_ns
    assert service.render_preview(clip.video_id, 6000, 14_000) == first
    assert first.stat().st_mtime_ns == stamp
    assert ffmpeg.probe(first).height == 640
