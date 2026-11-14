"""Vertical framing, clip export, thumbnails and subtitles."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from clipping_machine import ffmpeg
from clipping_machine.clipping import subtitle_burner
from clipping_machine.clipping.clip_builder import (
    RenderOptions,
    clip_filename,
    format_position,
    render_clip,
    unique_path,
)
from clipping_machine.clipping.subtitle_burner import Cue
from clipping_machine.clipping.thumbnail_generator import generate_thumbnail
from clipping_machine.clipping.vertical_formatter import (
    ContentBox,
    analyze_framing,
    blur_pad_filter,
    content_box_from_fractions,
    crop_filter,
    detect_content_box,
    estimate_focus_x,
    focus_from_frames,
    limit_focus,
    vertical_filter,
)
from clipping_machine.config import ExportConfig

# ------------------------------------------------------------- filter strings


def test_center_crop_of_a_landscape_source() -> None:
    # 1920x1080 scaled to 3414x1920, then a centred 1080-wide crop.
    assert crop_filter(1920, 1080, 1080, 1920) == "scale=3414:1920,crop=1080:1920:1167:0,setsar=1"


def test_crop_follows_the_focus_and_stays_in_frame() -> None:
    assert ":0:0,setsar=1" in crop_filter(1920, 1080, 1080, 1920, focus_x=0.0)
    assert "crop=1080:1920:2334:0" in crop_filter(1920, 1080, 1080, 1920, focus_x=1.0)
    assert crop_filter(1920, 1080, 1080, 1920, focus_x=7.0) == crop_filter(
        1920, 1080, 1080, 1920, focus_x=1.0
    )


def test_portrait_source_is_cropped_top_and_bottom() -> None:
    assert crop_filter(1080, 2400, 1080, 1920) == "scale=1080:2400,crop=1080:1920:0:240,setsar=1"
    assert crop_filter(1080, 1920, 1080, 1920) == "scale=1080:1920,crop=1080:1920:0:0,setsar=1"


def test_crop_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        crop_filter(0, 1080, 1080, 1920)


def test_vertical_filter_variants() -> None:
    assert vertical_filter(1920, 1080, 1080, 1920, framing="center", focus_x=0.9) == crop_filter(
        1920, 1080, 1080, 1920, 0.5
    )
    assert vertical_filter(1920, 1080, 1080, 1920, framing="smart", focus_x=0.9) == crop_filter(
        1920, 1080, 1080, 1920, 0.9
    )
    assert vertical_filter(1920, 1080, 1080, 1920, framing="blur_pad") == blur_pad_filter(
        1080, 1920
    )
    capped = vertical_filter(1920, 1080, 1080, 1920, src_fps=120, max_fps=60)
    assert capped.endswith(",fps=60")
    assert "fps=" not in vertical_filter(1920, 1080, 1080, 1920, src_fps=59.94, max_fps=60)


def test_focus_from_frames_locates_localized_motion() -> None:
    frames = np.zeros((6, 20, 40), dtype=np.uint8)
    for index in range(6):
        frames[index, :, 32:38] = 0 if index % 2 else 255  # flicker on the right
    assert focus_from_frames(frames) > 0.75
    mirrored = frames[:, :, ::-1].copy()
    assert focus_from_frames(mirrored) < 0.25


def test_smart_crop_keeps_the_screen_centre_in_frame() -> None:
    # 16:9 -> 9:16 keeps 31.6% of the width. With the default band (0.5) the source centre
    # must stay in the middle half of the crop: focus within 0.5 +/- 0.079.
    reach = (1080 / 1920) / (1920 / 1080) * 0.25
    assert limit_focus(0.77, 1920, 1080, 1080, 1920) == pytest.approx(0.5 + reach)
    assert limit_focus(0.10, 1920, 1080, 1080, 1920) == pytest.approx(0.5 - reach)
    assert limit_focus(0.52, 1920, 1080, 1080, 1920) == 0.52  # small shifts pass through
    assert limit_focus(0.9, 1920, 1080, 1080, 1920, center_band=0.0) == 0.5
    # Band 1: the centre may reach the crop edge, never leave it.
    assert limit_focus(1.0, 1920, 1080, 1080, 1920, center_band=1.0) == pytest.approx(
        0.5 + reach * 2
    )
    # A source that is not wider than 9:16 is not cropped sideways.
    assert limit_focus(0.9, 1080, 1920, 1080, 1920) == 0.5
    assert limit_focus(0.9, 0, 1080, 1080, 1920) == 0.5


def test_limited_focus_never_crops_out_the_centre() -> None:
    for focus in (0.0, 0.2, 0.5, 0.8, 1.0):
        safe = limit_focus(focus, 1920, 1080, 1080, 1920)
        chain = crop_filter(1920, 1080, 1080, 1920, safe)
        x = int(chain.split("crop=1080:1920:")[1].split(":")[0])
        centre = 3414 / 2  # source centre in the scaled frame
        assert x + 1080 * 0.25 <= centre <= x + 1080 * 0.75


def test_focus_defaults_to_centre_without_a_clear_subject() -> None:
    still = np.full((6, 20, 40), 90, dtype=np.uint8)
    assert focus_from_frames(still) == 0.5
    pan = np.stack([np.full((20, 40), value, dtype=np.uint8) for value in (0, 50, 100, 150)])
    assert focus_from_frames(pan) == 0.5  # whole-frame change is not a subject
    assert focus_from_frames(np.zeros((1, 20, 40), dtype=np.uint8)) == 0.5


# -------------------------------------------------------------- black borders


def _bordered(box: tuple[int, int, int, int] = (20, 5, 140, 85), frames: int = 4) -> np.ndarray:
    """Gray-level clip whose picture fills ``box`` (left, top, right, bottom); the rest is black."""
    rng = np.random.default_rng(0)
    clip = np.full((frames, 90, 160), 16, dtype=np.uint8)  # 160x90, video-range black
    left, top, right, bottom = box
    clip[:, top:bottom, left:right] = rng.integers(40, 230, (frames, bottom - top, right - left))
    return clip


def test_detects_pillarbox_and_letterbox() -> None:
    fractions = detect_content_box(_bordered())
    assert fractions is not None
    left, top, right, bottom = fractions
    # One analysis pixel inside each real border.
    assert (left, top, right, bottom) == (21 / 160, 6 / 90, 139 / 160, 84 / 90)


def test_full_frame_picture_is_left_alone() -> None:
    assert detect_content_box(_bordered(box=(0, 0, 160, 90))) is None
    # Borders thinner than 1% of the frame are noise, not letterboxing.
    assert detect_content_box(_bordered(box=(1, 0, 160, 90))) is None


def test_dark_moment_in_one_frame_is_not_a_border() -> None:
    clip = _bordered(box=(0, 0, 160, 90))
    clip[:3, :, :40] = 10  # the left side is dark for a while, but not for the whole clip
    assert detect_content_box(clip) is None


def test_small_overlay_inside_a_border_does_not_stop_the_crop() -> None:
    # A dim logo in the black side bar (as seen in a real capture) must not keep the bar.
    clip = _bordered(box=(20, 0, 140, 90))
    clip[:, 80:90, 145:158] = 60
    fractions = detect_content_box(clip)
    assert fractions is not None and fractions[2] == 139 / 160


def test_implausible_boxes_are_rejected() -> None:
    assert detect_content_box(np.full((4, 90, 160), 16, dtype=np.uint8)) is None  # all black
    assert detect_content_box(_bordered(box=(60, 0, 100, 90))) is None  # keeps < half the width
    assert detect_content_box(np.zeros((0, 90, 160), dtype=np.uint8)) is None


def test_content_box_rounds_inwards_to_even_pixels() -> None:
    box = content_box_from_fractions((21 / 160, 6 / 90, 139 / 160, 84 / 90), 1920, 1080)
    assert box == ContentBox(x=252, y=72, width=1416, height=936)
    assert all(value % 2 == 0 for value in (box.x, box.y, box.width, box.height))
    assert box.crop() == "crop=1416:936:252:72"


def test_vertical_filter_cuts_the_borders_first() -> None:
    box = ContentBox(240, 30, 1440, 1020)
    chain = vertical_filter(1920, 1080, 1080, 1920, framing="center", content_box=box)
    assert chain == "crop=1440:1020:240:30," + crop_filter(1440, 1020, 1080, 1920, 0.5)
    padded = vertical_filter(1920, 1080, 1080, 1920, framing="blur_pad", content_box=box)
    assert padded == "crop=1440:1020:240:30," + blur_pad_filter(1080, 1920)


# -------------------------------------------------------------------- naming


def test_clip_filename_pattern() -> None:
    name = clip_filename("gta6_clip", date(2026, 11, 14), 724_000, 0.93)
    assert name == "gta6_clip_2026-11-14_00-12-04_score093.mp4"
    assert clip_filename("x", date(2026, 1, 2), 0, 1.7).endswith("_00-00-00_score100.mp4")
    assert format_position(3_725_000) == "01-02-05"


def test_unique_path_never_overwrites(tmp_path: Path) -> None:
    first = tmp_path / "clip.mp4"
    assert unique_path(first) == first
    first.write_bytes(b"x")
    second = unique_path(first)
    assert second.name == "clip_2.mp4"
    second.write_bytes(b"x")
    assert unique_path(first).name == "clip_3.mp4"


def test_render_options_presets() -> None:
    config = ExportConfig()
    final = RenderOptions.final(config)
    assert (final.width, final.height, final.framing) == (1080, 1920, "smart")
    draft = RenderOptions.draft(config, framing="center")
    assert (draft.width, draft.height) == (360, 640)
    assert draft.preset == "ultrafast" and draft.framing == "center"


# --------------------------------------------------------------------- export


@pytest.mark.ffmpeg
@pytest.mark.parametrize("framing", ["center", "smart", "blur_pad"])
def test_export_is_a_vertical_h264_mp4(sample_video: Path, tmp_path: Path, framing: str) -> None:
    info = ffmpeg.probe(sample_video)
    options = RenderOptions.final(ExportConfig(preset="ultrafast"), framing=framing, focus_x=0.7)  # type: ignore[arg-type]
    progress: list[float] = []
    output = render_clip(
        sample_video,
        info,
        4000,
        9000,
        tmp_path / f"{framing}.mp4",
        options,
        on_progress=progress.append,
    )
    result = ffmpeg.probe(output)
    assert (result.width, result.height) == (1080, 1920)
    assert result.width * 16 == result.height * 9
    assert result.video_codec == "h264"
    assert result.audio_codec == "aac"
    assert abs(result.duration_ms - 5000) < 250
    assert "mp4" in result.format_name
    assert progress and progress[-1] == 1.0


@pytest.mark.ffmpeg
def test_export_of_a_silent_source_has_no_audio_track(silent_video: Path, tmp_path: Path) -> None:
    info = ffmpeg.probe(silent_video)
    options = RenderOptions.draft(ExportConfig(), framing="center")
    output = render_clip(silent_video, info, 0, 3000, tmp_path / "silent.mp4", options)
    result = ffmpeg.probe(output)
    assert (result.width, result.height) == (360, 640)
    assert result.audio_codec is None


@pytest.mark.ffmpeg
def test_export_clamps_the_end_and_validates_the_range(sample_video: Path, tmp_path: Path) -> None:
    info = ffmpeg.probe(sample_video)
    options = RenderOptions.draft(ExportConfig(), framing="center")
    output = render_clip(sample_video, info, 18_000, 99_000, tmp_path / "tail.mp4", options)
    assert abs(ffmpeg.probe(output).duration_ms - 2000) < 250
    with pytest.raises(ValueError):
        render_clip(sample_video, info, 5000, 5000, tmp_path / "x.mp4", options)
    with pytest.raises(ValueError):
        render_clip(sample_video, info, 30_000, 35_000, tmp_path / "y.mp4", options)


@pytest.mark.ffmpeg
def test_export_strips_source_metadata(tmp_path: Path) -> None:
    import subprocess

    source = tmp_path / "tagged.mp4"
    subprocess.run(
        [
            ffmpeg.ffmpeg_bin(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-t",
            "3",
            "-i",
            "testsrc2=size=320x180:rate=15",
            "-metadata",
            "title=My private recording title",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )
    output = render_clip(
        source,
        ffmpeg.probe(source),
        0,
        2000,
        tmp_path / "clean.mp4",
        RenderOptions.draft(ExportConfig(), framing="center"),
    )
    tags = subprocess.run(
        [ffmpeg.ffprobe_bin(), "-v", "error", "-show_entries", "format_tags", str(output)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "private recording" not in tags


@pytest.mark.ffmpeg
def test_estimate_focus_runs_on_real_video(sample_video: Path) -> None:
    focus = estimate_focus_x(sample_video, ffmpeg.probe(sample_video), 2000, 6000)
    assert 0.1 <= focus <= 0.9


@pytest.mark.ffmpeg
def test_recorded_black_bars_do_not_reach_the_vertical_clip(tmp_path: Path) -> None:
    from PIL import Image

    # A 4:3-ish picture inside a 16:9 recording: black bars on all four sides.
    source = tmp_path / "bordered.mp4"
    ffmpeg.run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-t",
            "3",
            "-i",
            "testsrc2=size=360x300:rate=15",
            "-vf",
            "pad=640:360:140:30:black",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ]
    )
    info = ffmpeg.probe(source)
    analysis = analyze_framing(source, info, 0, 3000, detect_borders=True, track_focus=True)
    box = analysis.content_box
    assert box is not None
    assert 140 <= box.x <= 150 and 30 <= box.y <= 40
    assert 490 <= box.x + box.width <= 500 and 320 <= box.y + box.height <= 330
    assert analysis.focus_x is not None and 0.0 <= analysis.focus_x <= 1.0

    output = render_clip(
        source,
        info,
        0,
        2000,
        tmp_path / "clip.mp4",
        RenderOptions.draft(ExportConfig(), framing="blur_pad", content_box=box),
    )
    still = tmp_path / "frame.png"
    ffmpeg.run_ffmpeg(["-ss", "1", "-i", str(output), "-frames:v", "1", str(still)])
    with Image.open(still) as image:
        pixels = np.asarray(image.convert("L"), dtype=np.float32)
    # The foreground spans the full width, so its edge columns are picture, not bar.
    middle = pixels[pixels.shape[0] // 2 - 20 : pixels.shape[0] // 2 + 20]
    assert middle[:, :3].mean() > 30 and middle[:, -3:].mean() > 30

    # Nothing to cut in a full-frame recording.
    plain = analyze_framing(source, info, 0, 3000, detect_borders=False, track_focus=False)
    assert plain.content_box is None and plain.focus_x is None


@pytest.mark.ffmpeg
def test_thumbnail_matches_the_clip_framing(sample_video: Path, tmp_path: Path) -> None:
    from PIL import Image

    output = generate_thumbnail(
        sample_video,
        ffmpeg.probe(sample_video),
        5000,
        tmp_path / "thumb.jpg",
        width=1080,
        height=1920,
        framing="center",
    )
    with Image.open(output) as image:
        assert image.size == (1080, 1920)
    # A timestamp past the end is clamped instead of failing.
    late = generate_thumbnail(
        sample_video,
        ffmpeg.probe(sample_video),
        999_000,
        tmp_path / "late.jpg",
        width=360,
        height=640,
    )
    assert late.stat().st_size > 0


# ------------------------------------------------------------------ subtitles


def test_srt_formatting() -> None:
    text = subtitle_burner.format_srt(
        [
            Cue(3_500, 5_250, "second line"),
            Cue(0, 1_500, "first line"),
            Cue(6_000, 6_000, "zero length is dropped"),
            Cue(7_000, 8_000, "   "),
        ]
    )
    assert text == (
        "1\n00:00:00,000 --> 00:00:01,500\nfirst line\n\n"
        "2\n00:00:03,500 --> 00:00:05,250\nsecond line\n"
    )
    assert subtitle_burner.format_timestamp(3_725_042) == "01:02:05,042"


def test_long_cues_are_wrapped_for_a_vertical_frame() -> None:
    wrapped = subtitle_burner.wrap_text(
        "this is a rather long sentence spoken in the gameplay clip"
    )
    lines = wrapped.split("\n")
    assert len(lines) == 2
    assert len(lines[0]) <= 28


def test_write_srt_and_filter(tmp_path: Path) -> None:
    assert subtitle_burner.write_srt([], tmp_path) is None
    path = subtitle_burner.write_srt([Cue(0, 1000, "olá, mundo")], tmp_path / "work")
    assert path is not None and path.name == "subtitles.srt"
    assert "olá, mundo" in path.read_text(encoding="utf-8")
    assert subtitle_burner.subtitles_filter().startswith("subtitles=filename=subtitles.srt:")
    with pytest.raises(ValueError):
        subtitle_burner.subtitles_filter(r"C:\work\subtitles.srt")


@pytest.mark.ffmpeg
def test_export_with_burned_subtitles(sample_video: Path, tmp_path: Path) -> None:
    srt = subtitle_burner.write_srt([Cue(0, 2500, "Legenda de teste")], tmp_path / "with space")
    assert srt is not None
    info = ffmpeg.probe(sample_video)
    options = RenderOptions.draft(ExportConfig(), framing="center")
    plain = render_clip(sample_video, info, 0, 3000, tmp_path / "plain.mp4", options)
    burned = render_clip(
        sample_video, info, 0, 3000, tmp_path / "burned.mp4", options, subtitles_path=srt
    )
    assert ffmpeg.probe(burned).duration_ms == pytest.approx(3000, abs=250)

    def frame(path: Path) -> np.ndarray:
        return next(ffmpeg.iter_gray_frames(path, fps=1, width=180, height=320, start_ms=1000))

    # The subtitle changes pixels in the lower part of the frame only.
    difference = np.abs(frame(plain).astype(int) - frame(burned).astype(int))
    assert difference[200:, :].mean() > 4 * max(difference[:100, :].mean(), 0.05)
