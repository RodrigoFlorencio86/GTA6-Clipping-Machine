"""Validation, import, windows, keyframes, audio and scene extraction."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from clipping_machine import ffmpeg
from clipping_machine.config import SceneConfig, Settings, UploadConfig, VideoConfig
from clipping_machine.ingest import scene_splitter
from clipping_machine.ingest.audio_chunker import AudioTrack, extract_audio, iter_chunks
from clipping_machine.ingest.frame_extractor import extract_frames, motion_from_frames
from clipping_machine.ingest.video_importer import (
    DuplicateVideoError,
    RightsNotConfirmedError,
    VideoImporter,
    fingerprint,
    safe_filename,
)
from clipping_machine.ingest.video_validator import (
    VideoValidationError,
    looks_like_url,
    validate_source,
    validate_video,
)
from clipping_machine.ingest.windows import (
    Window,
    compute_windows,
    keyframe_times_ms,
    keyframes_for_window,
)
from clipping_machine.storage.database import Database
from clipping_machine.storage.models import VideoStatus
from clipping_machine.storage.repositories import VideoRepository

# ----------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "value",
    [
        "https://www.youtube.com/watch?v=abc",
        "http://example.com/video.mp4",
        "ftp://host/video.mp4",
        "rtmp://live.twitch.tv/app/key",
        "www.tiktok.com/@user/video/1",
        "youtu.be/abc",
        "twitch.tv/videos/123",
    ],
)
def test_urls_are_recognised(value: str) -> None:
    assert looks_like_url(value)


@pytest.mark.parametrize(
    "value", [r"C:\Videos\gameplay.mp4", "/home/me/gameplay.mkv", "gameplay.mp4", r"D:\a b\c.mov"]
)
def test_local_paths_are_not_urls(value: str) -> None:
    assert not looks_like_url(value)


def test_links_are_rejected_before_touching_the_network() -> None:
    issues = validate_source("https://www.youtube.com/watch?v=abc", UploadConfig())
    assert [issue.code for issue in issues] == ["url_not_supported"]


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    issues = validate_source(str(tmp_path / "nope.mp4"), UploadConfig())
    assert [issue.code for issue in issues] == ["not_found"]


def test_wrong_extension_and_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes(b"")
    codes = {issue.code for issue in validate_source(str(path), UploadConfig())}
    assert codes == {"unsupported_extension", "empty_file"}


def test_size_limit(tmp_path: Path) -> None:
    path = tmp_path / "big.mp4"
    path.write_bytes(b"0" * (2 * 1024 * 1024))
    codes = [i.code for i in validate_source(str(path), UploadConfig(max_file_size_mb=1))]
    assert codes == ["file_too_large"]


@pytest.mark.ffmpeg
def test_garbage_with_video_extension_is_unreadable(tmp_path: Path) -> None:
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"this is not a video" * 100)
    result = validate_video(path, UploadConfig())
    assert not result.ok
    assert [issue.code for issue in result.issues] == ["unreadable"]


@pytest.mark.ffmpeg
def test_valid_video_metadata(sample_video: Path) -> None:
    result = validate_video(sample_video, UploadConfig(min_width=160, min_height=90))
    assert result.ok and result.info is not None
    info = result.info
    assert (info.width, info.height) == (320, 180)
    assert abs(info.duration_ms - 20_000) < 300
    assert info.video_codec == "h264"
    assert info.audio_codec == "aac"
    assert abs(info.fps - 15) < 0.1
    assert info.size_bytes == sample_video.stat().st_size


@pytest.mark.ffmpeg
def test_duration_and_resolution_limits(sample_video: Path) -> None:
    config = UploadConfig(max_video_duration_minutes=0.1, min_width=1280, min_height=720)
    codes = {issue.code for issue in validate_video(sample_video, config).issues}
    assert codes == {"too_long", "resolution_too_low"}


# --------------------------------------------------------------------- import


def _importer(settings: Settings) -> tuple[VideoImporter, VideoRepository]:
    settings.ensure_dirs()
    videos = VideoRepository(Database(settings.db_path))
    return VideoImporter(settings, videos), videos


def test_safe_filename_strips_directories_and_odd_characters() -> None:
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename(r"C:\Users\me\my clip (1).mp4") == "my_clip_1_.mp4"
    assert safe_filename("...") == "video"


def test_import_requires_rights_confirmation(settings: Settings, tmp_path: Path) -> None:
    importer, videos = _importer(settings)
    with pytest.raises(RightsNotConfirmedError):
        importer.import_path(str(tmp_path / "x.mp4"), rights_confirmed=False)
    with pytest.raises(RightsNotConfirmedError):
        importer.import_upload(io.BytesIO(b""), "x.mp4", rights_confirmed=False)
    assert videos.all() == []


def test_import_rejects_links(settings: Settings) -> None:
    importer, _ = _importer(settings)
    with pytest.raises(VideoValidationError) as error:
        importer.import_path("https://www.twitch.tv/videos/1", rights_confirmed=True)
    assert error.value.issues[0].code == "url_not_supported"


@pytest.mark.ffmpeg
def test_import_path_in_place(settings: Settings, sample_video: Path) -> None:
    importer, videos = _importer(settings)
    video = importer.import_path(str(sample_video), rights_confirmed=True)
    assert video.status is VideoStatus.UPLOADED
    assert video.managed_copy is False
    assert Path(video.path) == sample_video.resolve()
    assert video.rights_confirmed_at
    assert videos.require(video.video_id).duration_ms == video.duration_ms
    assert list(settings.uploads_dir.iterdir()) == []


@pytest.mark.ffmpeg
def test_import_upload_saves_a_managed_copy(settings: Settings, sample_video: Path) -> None:
    importer, _ = _importer(settings)
    with sample_video.open("rb") as stream:
        video = importer.import_upload(stream, "../My Gameplay.mp4", rights_confirmed=True)
    stored = Path(video.path)
    assert video.managed_copy is True
    assert stored.parent == settings.uploads_dir
    assert stored.read_bytes() == sample_video.read_bytes()
    assert video.original_name == "My_Gameplay.mp4"


@pytest.mark.ffmpeg
def test_duplicate_import_is_detected(settings: Settings, sample_video: Path) -> None:
    importer, _ = _importer(settings)
    first = importer.import_path(str(sample_video), rights_confirmed=True)
    with pytest.raises(DuplicateVideoError) as error:
        importer.import_path(str(sample_video), rights_confirmed=True, copy=True)
    assert error.value.existing.video_id == first.video_id
    assert list(settings.uploads_dir.iterdir()) == []  # the rejected copy was cleaned up


@pytest.mark.ffmpeg
def test_invalid_upload_leaves_no_file_behind(settings: Settings) -> None:
    importer, videos = _importer(settings)
    with pytest.raises(VideoValidationError):
        importer.import_upload(io.BytesIO(b"junk" * 500), "clip.mp4", rights_confirmed=True)
    assert list(settings.uploads_dir.iterdir()) == []
    assert videos.all() == []


def test_fingerprint_depends_on_content(tmp_path: Path) -> None:
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"a" * 5000)
    b.write_bytes(b"a" * 4999 + b"b")
    assert fingerprint(a) == fingerprint(a)
    assert fingerprint(a) != fingerprint(b)


# -------------------------------------------------------------------- windows


def test_windows_follow_the_spec_defaults() -> None:
    windows = compute_windows(30_000, 8_000, 2_000)
    assert [(w.start_ms, w.end_ms) for w in windows] == [
        (0, 8000),
        (6000, 14000),
        (12000, 20000),
        (18000, 26000),
        (24000, 30000),
    ]
    assert [w.index for w in windows] == [0, 1, 2, 3, 4]


def test_windows_cover_the_whole_video_without_redundant_tail() -> None:
    windows = compute_windows(14_000, 8_000, 2_000)
    assert [(w.start_ms, w.end_ms) for w in windows] == [(0, 8000), (6000, 14000)]
    assert compute_windows(5_000, 8_000, 2_000) == [Window(0, 0, 5000)]
    assert compute_windows(0, 8_000, 2_000) == []


def test_windows_reject_bad_parameters() -> None:
    with pytest.raises(ValueError):
        compute_windows(10_000, 8_000, 8_000)
    with pytest.raises(ValueError):
        compute_windows(10_000, 0, 0)


def test_keyframe_grid_and_window_membership() -> None:
    assert keyframe_times_ms(9_000, 2_000) == [0, 2000, 4000, 6000, 8000]
    assert keyframes_for_window(Window(1, 6000, 14000), 2000, 15) == [3, 4, 5, 6]
    assert keyframes_for_window(Window(0, 0, 8000), 2000, 15) == [0, 1, 2, 3]
    # Window past the last keyframe falls back to the nearest one.
    assert keyframes_for_window(Window(9, 29_000, 29_500), 2000, 15) == [14]
    assert keyframes_for_window(Window(0, 0, 8000), 2000, 0) == []


# --------------------------------------------------------- frames and motion


def test_motion_from_frames() -> None:
    still = np.full((3, 4, 4), 100, dtype=np.uint8)
    assert motion_from_frames(still).tolist() == [0.0, 0.0, 0.0]
    moving = np.stack([np.full((4, 4), value, dtype=np.uint8) for value in (0, 10, 40)])
    assert motion_from_frames(moving).tolist() == [0.0, 10.0, 30.0]
    assert motion_from_frames(np.zeros((0, 4, 4), dtype=np.uint8)).size == 0


@pytest.mark.ffmpeg
def test_extract_keyframes_and_motion(sample_video: Path, tmp_path: Path) -> None:
    info = ffmpeg.probe(sample_video)
    progress: list[float] = []
    result = extract_frames(
        sample_video, tmp_path / "frames", info, VideoConfig(), on_progress=progress.append
    )
    # 20 s at one keyframe every 2 s.
    assert len(result.keyframes) == 10
    assert result.keyframes[0].name == "kf_000000.jpg"
    assert all(path.stat().st_size > 0 for path in result.keyframes)
    assert result.keyframe_time_ms(3) == 6000
    # 4 motion samples per second.
    assert abs(result.motion.size - 80) <= 2
    assert result.motion[0] == 0.0
    assert result.motion_between(0, 8000) > 0  # testsrc2 animates
    assert progress[-1] == 1.0

    from PIL import Image

    with Image.open(result.keyframes[0]) as image:
        assert image.size == (320, 180)  # never upscaled


@pytest.mark.ffmpeg
def test_keyframes_are_downscaled_to_the_configured_side(
    sample_video: Path, tmp_path: Path
) -> None:
    info = ffmpeg.probe(sample_video)
    result = extract_frames(
        sample_video, tmp_path / "small", info, VideoConfig(keyframe_max_side=128)
    )
    from PIL import Image

    with Image.open(result.keyframes[0]) as image:
        assert max(image.size) == 128
        assert image.size[0] % 2 == 0 and image.size[1] % 2 == 0


@pytest.mark.ffmpeg
def test_reextraction_replaces_old_keyframes(sample_video: Path, tmp_path: Path) -> None:
    target = tmp_path / "frames"
    target.mkdir()
    (target / "kf_999999.jpg").write_bytes(b"stale")
    result = extract_frames(sample_video, target, ffmpeg.probe(sample_video), VideoConfig())
    assert not (target / "kf_999999.jpg").exists()
    assert len(result.keyframes) == 10


# ---------------------------------------------------------------------- audio


@pytest.mark.ffmpeg
def test_extract_audio_and_chunks(sample_video: Path, tmp_path: Path) -> None:
    wav = extract_audio(sample_video, tmp_path / "audio.wav", 16000)
    windows = compute_windows(20_000, 8_000, 2_000)
    with AudioTrack(wav) as track:
        assert track.sample_rate == 16000
        assert abs(track.duration_ms - 20_000) < 200
        chunks = list(iter_chunks(track, windows))
        assert len(chunks) == len(windows)
        first_window, first = chunks[0]
        assert first_window == windows[0]
        assert first.dtype == np.float32
        assert first.size == 8 * 16000
        assert float(np.abs(first).max()) <= 1.0
        # The fixture is quiet for 10 s, then loud.
        quiet = float(np.sqrt(np.mean(track.read(1000, 9000) ** 2)))
        loud = float(np.sqrt(np.mean(track.read(11_000, 19_000) ** 2)))
        assert loud > 10 * quiet
        assert track.read(25_000, 30_000).size == 0


@pytest.mark.ffmpeg
def test_extract_audio_range(sample_video: Path, tmp_path: Path) -> None:
    wav = extract_audio(sample_video, tmp_path / "part.wav", 16000, start_ms=5000, clip_ms=3000)
    with AudioTrack(wav) as track:
        assert abs(track.duration_ms - 3000) < 150


@pytest.mark.ffmpeg
def test_extract_audio_fails_without_audio_stream(silent_video: Path, tmp_path: Path) -> None:
    with pytest.raises(ffmpeg.FFmpegError):
        extract_audio(silent_video, tmp_path / "none.wav", 16000)


# --------------------------------------------------------------------- scenes


def test_scenes_from_boundaries_are_contiguous() -> None:
    scenes = scene_splitter.scenes_from_boundaries("v", [(0, 4000), (4000, 9000)], 10_000)
    assert [(s.start_ms, s.end_ms) for s in scenes] == [(0, 4000), (4000, 10_000)]
    assert [s.scene_index for s in scenes] == [0, 1]
    single = scene_splitter.scenes_from_boundaries("v", [], 10_000)
    assert [(s.start_ms, s.end_ms) for s in single] == [(0, 10_000)]


def test_cut_helpers() -> None:
    scenes = scene_splitter.scenes_from_boundaries("v", [(0, 4000), (4000, 9000)], 10_000)
    cuts = scene_splitter.cut_times_ms(scenes)
    assert cuts == [4000]
    assert scene_splitter.count_cuts(cuts, 0, 8000) == 1
    assert scene_splitter.count_cuts(cuts, 4000, 8000) == 0  # a cut on the edge is not inside
    assert scene_splitter.snap_to_cut(4600, cuts, 1000) == 4000
    assert scene_splitter.snap_to_cut(6000, cuts, 1000) == 6000
    assert scene_splitter.snap_to_cut(6000, [], 1000) == 6000


@pytest.mark.ffmpeg
def test_detect_scenes_finds_the_hard_cut(two_scene_video: Path) -> None:
    info = ffmpeg.probe(two_scene_video)
    scenes = scene_splitter.detect_scenes(
        two_scene_video, "vid", info.duration_ms, info.fps, SceneConfig(frame_skip=0)
    )
    assert len(scenes) == 2
    assert abs(scenes[1].start_ms - 4000) <= 200
    assert scenes[0].start_ms == 0 and scenes[-1].end_ms == info.duration_ms


@pytest.mark.ffmpeg
def test_detect_scenes_single_scene(sample_video: Path) -> None:
    info = ffmpeg.probe(sample_video)
    scenes = scene_splitter.detect_scenes(
        sample_video, "vid", info.duration_ms, info.fps, SceneConfig(threshold=80)
    )
    assert [(s.start_ms, s.end_ms) for s in scenes] == [(0, info.duration_ms)]
