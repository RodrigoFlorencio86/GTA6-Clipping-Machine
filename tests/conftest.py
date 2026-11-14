"""Shared fixtures.

The suite never loads the real embedding model: ``FakeEmbedder`` (tests/fakes.py)
stands in for it. Tests that need FFmpeg are skipped when it is not installed.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from clipping_machine import ffmpeg
from clipping_machine.config import Settings, load_settings
from clipping_machine.services import ClippingService
from tests.fakes import FakeEmbedder

FFMPEG_AVAILABLE = ffmpeg.is_available()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    skip = pytest.mark.skip(reason="ffmpeg/ffprobe not installed")
    for item in items:
        if "ffmpeg" in item.keywords and not FFMPEG_AVAILABLE:
            item.add_marker(skip)


def make_video(
    path: Path,
    *,
    seconds: float = 20.0,
    size: str = "320x180",
    fps: int = 15,
    audio: bool = True,
    two_scenes: bool = False,
) -> Path:
    """Generate a small synthetic video with FFmpeg's test sources."""
    command = [ffmpeg.ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y"]
    if two_scenes:
        half = seconds / 2
        command += [
            "-f",
            "lavfi",
            "-t",
            f"{half}",
            "-i",
            f"testsrc2=size={size}:rate={fps}",
            "-f",
            "lavfi",
            "-t",
            f"{half}",
            "-i",
            f"color=c=red:size={size}:rate={fps}",
        ]
        video_map = ["-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]"]
        audio_index = 2
    else:
        command += ["-f", "lavfi", "-t", f"{seconds}", "-i", f"testsrc2=size={size}:rate={fps}"]
        video_map = ["-map", "0:v"]
        audio_index = 1
    if audio:
        # Quiet first half, loud second half.
        volume = f"volume='if(lt(t,{seconds / 2}),0.02,0.8)':eval=frame"
        command += [
            "-f",
            "lavfi",
            "-t",
            f"{seconds}",
            "-i",
            f"sine=frequency=440:sample_rate=44100,{volume}",
        ]
    command += video_map
    if audio:
        command += ["-map", f"{audio_index}:a", "-c:a", "aac", "-b:a", "64k"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """20 s, 320x180, with audio (quiet then loud)."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg/ffprobe not installed")
    return make_video(tmp_path_factory.mktemp("media") / "gameplay.mp4")


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg/ffprobe not installed")
    return make_video(tmp_path_factory.mktemp("media") / "silent.mkv", seconds=10, audio=False)


@pytest.fixture(scope="session")
def two_scene_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg/ffprobe not installed")
    return make_video(
        tmp_path_factory.mktemp("media") / "scenes.mp4", seconds=8, audio=False, two_scenes=True
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Repository configs with an isolated data directory."""
    return load_settings(env={"GCM_DATA_DIR": str(tmp_path / "data")})


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def service(settings: Settings, embedder: FakeEmbedder) -> Iterator[ClippingService]:
    instance = ClippingService(settings, backend=embedder)
    yield instance
    instance.jobs.shutdown()
