"""Scene (shot) detection with PySceneDetect."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from clipping_machine.config import SceneConfig
from clipping_machine.storage.models import Scene


def detect_scenes(
    video_path: Path, video_id: str, duration_ms: int, fps: float, config: SceneConfig
) -> list[Scene]:
    """Return contiguous scenes covering the whole video.

    Continuous gameplay often has no hard cuts; the result is then one scene.
    """
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    manager = SceneManager()
    min_scene_frames = max(1, round(config.min_scene_seconds * (fps or 30.0)))
    manager.add_detector(
        ContentDetector(threshold=config.threshold, min_scene_len=min_scene_frames)
    )
    manager.detect_scenes(video=video, frame_skip=config.frame_skip, show_progress=False)
    raw = manager.get_scene_list(start_in_scene=True)

    boundaries = [
        (round(start.get_seconds() * 1000), round(end.get_seconds() * 1000)) for start, end in raw
    ]
    return scenes_from_boundaries(video_id, boundaries, duration_ms)


def scenes_from_boundaries(
    video_id: str, boundaries: Sequence[tuple[int, int]], duration_ms: int
) -> list[Scene]:
    """Normalize detector output into gap-free scenes spanning ``[0, duration_ms]``."""
    cuts = sorted({start for start, _ in boundaries if 0 < start < duration_ms})
    edges = [0, *cuts, duration_ms]
    return [
        Scene(video_id, index, edges[index], edges[index + 1])
        for index in range(len(edges) - 1)
        if edges[index + 1] > edges[index]
    ]


def cut_times_ms(scenes: Sequence[Scene]) -> list[int]:
    """Timestamps where one scene ends and the next begins."""
    return [scene.start_ms for scene in scenes if scene.start_ms > 0]


def count_cuts(cuts_ms: Sequence[int], start_ms: int, end_ms: int) -> int:
    return sum(1 for cut in cuts_ms if start_ms < cut < end_ms)


def snap_to_cut(time_ms: int, cuts_ms: Sequence[int], tolerance_ms: int) -> int:
    """Move ``time_ms`` onto the nearest scene cut if one is within ``tolerance_ms``."""
    if not cuts_ms or tolerance_ms <= 0:
        return time_ms
    nearest = min(cuts_ms, key=lambda cut: abs(cut - time_ms))
    return nearest if abs(nearest - time_ms) <= tolerance_ms else time_ms
