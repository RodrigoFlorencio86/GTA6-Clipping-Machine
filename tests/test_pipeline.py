"""Indexing pipeline end to end (real FFmpeg and LanceDB, fake embedding model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from clipping_machine.embeddings.base import WindowInput
from clipping_machine.embeddings.video_embedder import embed_in_batches
from clipping_machine.pipeline.indexer import STAGES, segment_id_for
from clipping_machine.pipeline.jobs import JobManager
from clipping_machine.services import ClippingService
from clipping_machine.storage.models import VideoStatus
from tests.fakes import FakeEmbedder

pytestmark = pytest.mark.ffmpeg


def test_stage_weights_sum_to_one() -> None:
    assert sum(weight for _, weight in STAGES) == pytest.approx(1.0)


def test_index_video_creates_segments_vectors_and_previews(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    video = service.import_path(str(sample_video), rights_confirmed=True)
    stages: list[str] = []
    count = service.indexer.index_video(
        video.video_id, on_progress=lambda stage, _overall: stages.append(stage)
    )

    # 20 s with 8 s windows and 2 s overlap -> starts at 0, 6, 12.
    assert count == 3
    segments = service.segments.for_video(video.video_id)
    assert [(s.start_ms, s.end_ms) for s in segments] == [
        (0, 8000),
        (6000, 14_000),
        (12_000, 20_000),
    ]
    assert [s.segment_id for s in segments] == [segment_id_for(video.video_id, i) for i in range(3)]
    assert all(
        s.embedding_model == "fake/embedder" and s.embedding_dimension == 768 for s in segments
    )
    assert all(Path(s.preview_frame_path).is_file() for s in segments)
    assert service.store.count(video.video_id) == 3

    # Each window got its four keyframes (one every 2 s) and 8 s of 16 kHz audio.
    assert [len(call.frames) for call in embedder.window_calls] == [4, 4, 4]
    assert all(
        call.audio is not None and call.audio.size == 8 * 16000 for call in embedder.window_calls
    )
    assert all(call.sample_rate == 16000 for call in embedder.window_calls)

    # The fixture's audio is quiet for 10 s and loud afterwards.
    assert segments[0].audio_energy < segments[2].audio_energy
    assert segments[2].audio_energy == pytest.approx(1.0, abs=0.1)
    assert all(0.0 <= s.motion <= 1.0 and s.motion_raw > 0 for s in segments)

    stored = service.videos.require(video.video_id)
    assert stored.status is VideoStatus.INDEXED
    assert stored.progress == 1.0 and stored.segments_count == 3
    assert stored.embedding_model == "fake/embedder"
    assert stored.index_seconds is not None and stored.index_seconds > 0
    assert [name for name, _ in STAGES] == list(dict.fromkeys(stages))

    # Scratch files are removed; only previews stay.
    assert not (service.settings.processed_dir / video.video_id).exists()
    assert len(service.scenes.for_video(video.video_id)) >= 1


def test_video_without_audio_is_indexed_frames_only(
    service: ClippingService, embedder: FakeEmbedder, silent_video: Path
) -> None:
    video = service.import_path(str(silent_video), rights_confirmed=True)
    assert service.index_now(video.video_id) == 2
    assert all(call.audio is None for call in embedder.window_calls)
    assert all(s.audio_energy == 0.0 for s in service.segments.for_video(video.video_id))


def test_reindexing_replaces_previous_data(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    video = service.import_path(str(sample_video), rights_confirmed=True)
    service.index_now(video.video_id)
    embedder.reset()
    service.index_now(video.video_id)
    assert service.store.count(video.video_id) == 3
    assert len(service.segments.for_video(video.video_id)) == 3


def test_indexed_video_is_searchable(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    embedder.window_concepts = {2: "explosion"}
    video = service.import_path(str(sample_video), rights_confirmed=True)
    service.index_now(video.video_id)
    result = service.search("explosion", fill_to_min_results=False)
    assert [(m.start_ms, m.end_ms) for m in result.moments] == [(12_000, 20_000)]
    assert result.moments[0].video_id == video.video_id
    assert Path(result.moments[0].preview_frame_path).is_file()


def test_failure_is_recorded_on_the_video(service: ClippingService, sample_video: Path) -> None:
    video = service.import_path(str(sample_video), rights_confirmed=True)

    def boom(_windows: object) -> None:
        raise RuntimeError("model exploded")

    service.backend.embed_windows = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        service.index_now(video.video_id)
    stored = service.videos.require(video.video_id)
    assert stored.status is VideoStatus.FAILED
    assert "model exploded" in (stored.error or "")
    assert service.store.count(video.video_id) == 0


def test_missing_file_fails_cleanly(service: ClippingService, tmp_path: Path) -> None:
    from tests.conftest import make_video

    path = make_video(tmp_path / "gone.mp4", seconds=4, audio=False)
    video = service.import_path(str(path), rights_confirmed=True)
    path.unlink()
    with pytest.raises(Exception, match="missing"):
        service.index_now(video.video_id)
    assert service.videos.require(video.video_id).status is VideoStatus.FAILED


def test_background_job_runs_to_completion(service: ClippingService, sample_video: Path) -> None:
    video = service.import_path(str(sample_video), rights_confirmed=True)
    assert service.start_indexing(video.video_id) is True
    service.jobs.wait(video.video_id, timeout=120)
    assert service.videos.require(video.video_id).status is VideoStatus.INDEXED
    assert service.jobs.is_active(video.video_id) is False


def test_job_manager_does_not_queue_the_same_video_twice() -> None:
    import threading

    release = threading.Event()
    manager = JobManager(lambda _video_id: release.wait(10))
    try:
        assert manager.submit("vid_1") is True
        assert manager.submit("vid_1") is False
        assert manager.active_ids() == ["vid_1"]
        release.set()
        manager.wait("vid_1", timeout=10)
        assert manager.submit("vid_1") is True
    finally:
        release.set()
        manager.shutdown()


def test_interrupted_indexing_is_flagged_on_restart(
    service: ClippingService, sample_video: Path
) -> None:
    video = service.import_path(str(sample_video), rights_confirmed=True)
    service.videos.mark_indexing(video.video_id)  # simulate a crash mid-run
    restarted = ClippingService(service.settings, backend=FakeEmbedder())
    try:
        stored = restarted.videos.require(video.video_id)
        assert stored.status is VideoStatus.FAILED
        assert "interrupted" in (stored.error or "")
    finally:
        restarted.jobs.shutdown()


def test_delete_video_erases_everything_derived(
    service: ClippingService, embedder: FakeEmbedder, sample_video: Path
) -> None:
    embedder.window_concepts = {1: "chase"}
    with sample_video.open("rb") as stream:
        video = service.import_upload(stream, "mine.mp4", rights_confirmed=True)
    service.index_now(video.video_id)
    moment = service.search("chase", fill_to_min_results=False).moments[0]
    clip = service.create_clip(moment, query="chase", reason="r")
    draft = service.render_draft(clip.clip_id)
    assert draft.is_file()

    service.delete_video(video.video_id)
    assert service.videos.get(video.video_id) is None
    assert service.segments.for_video(video.video_id) == []
    assert service.clips.find(video_id=video.video_id) == []
    assert service.store.count(video.video_id) == 0
    assert not Path(video.path).exists()  # managed copy removed
    assert not draft.exists()
    assert not (service.settings.previews_dir / video.video_id).exists()


def test_delete_keeps_the_original_when_indexed_in_place(
    service: ClippingService, sample_video: Path
) -> None:
    video = service.import_path(str(sample_video), rights_confirmed=True)
    service.delete_video(video.video_id)
    assert sample_video.is_file()


def test_embed_in_batches_reports_progress() -> None:
    backend = FakeEmbedder(dimension=16)
    inputs = [WindowInput(frames=(Path("f.jpg"),)) for _ in range(5)]
    seen: list[float] = []
    vectors = embed_in_batches(backend, inputs, batch_size=2, on_progress=seen.append)
    assert vectors.shape == (5, 16)
    assert seen == [0.4, 0.8, 1.0]
    assert embed_in_batches(backend, [], batch_size=2).shape == (0, 16)
