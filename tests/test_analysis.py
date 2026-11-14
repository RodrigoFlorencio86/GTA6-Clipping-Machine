"""Audio loudness, normalization, zero-shot tags and the editorial score."""

from __future__ import annotations

import numpy as np
import pytest

from clipping_machine.analysis import audio_analyzer
from clipping_machine.analysis.highlight_scorer import (
    ScoreSignals,
    compute_score,
    duration_fit,
    novelty,
    temporal_overlap,
)
from clipping_machine.analysis.normalization import clamp01, robust_normalize, zscores
from clipping_machine.analysis.scene_describer import (
    SceneDescriber,
    reaction_presence,
    select_tags,
)
from clipping_machine.config import ScoreWeights, TagConfig, TagDefinition
from tests.fakes import CONCEPTS, FakeEmbedder

SR = 16000


def _tone(amplitude: float, seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------- audio


def test_full_scale_sine_is_about_minus_3_dbfs() -> None:
    stats = audio_analyzer.measure(_tone(1.0), SR)
    assert stats.mean_db == pytest.approx(-3.01, abs=0.1)
    assert stats.peak_db == pytest.approx(-3.01, abs=0.2)


def test_silence_and_empty_audio() -> None:
    assert audio_analyzer.measure(np.zeros(SR, dtype=np.float32), SR) == audio_analyzer.SILENT
    assert audio_analyzer.measure(np.zeros(0, dtype=np.float32), SR) == audio_analyzer.SILENT


def test_peak_captures_a_short_burst() -> None:
    samples = np.concatenate([_tone(0.01, 0.9), _tone(0.9, 0.1)])
    stats = audio_analyzer.measure(samples, SR)
    assert stats.peak_db > stats.mean_db + 5


def test_short_time_rms_frames() -> None:
    frames = audio_analyzer.short_time_rms(_tone(0.5, 1.0), SR, frame_ms=50)
    assert frames.size == 20
    assert np.allclose(frames, 0.5 / np.sqrt(2), atol=0.01)


def test_normalize_intensity_ranks_loud_above_quiet() -> None:
    stats = [audio_analyzer.measure(_tone(a), SR) for a in (0.01, 0.02, 0.05, 0.2, 0.9)]
    normalized = audio_analyzer.normalize_intensity(stats)
    assert normalized.tolist() == sorted(normalized.tolist())
    assert normalized[0] == pytest.approx(0.0, abs=0.05)
    assert normalized[-1] == pytest.approx(1.0, abs=0.05)


def test_normalize_intensity_of_a_silent_or_flat_video_is_zero() -> None:
    assert audio_analyzer.normalize_intensity([audio_analyzer.SILENT] * 4).tolist() == [0.0] * 4
    flat = [audio_analyzer.measure(_tone(0.3), SR)] * 4
    assert audio_analyzer.normalize_intensity(flat).tolist() == [0.0] * 4
    assert audio_analyzer.normalize_intensity([]).size == 0


# -------------------------------------------------------------- normalization


def test_robust_normalize_saturates_outliers() -> None:
    values = [*range(100), 10_000]
    normalized = robust_normalize(values)
    assert normalized.min() == 0.0 and normalized.max() == 1.0
    assert normalized[50] == pytest.approx(0.5, abs=0.06)  # the outlier did not squash the rest


def test_robust_normalize_flat_input_is_zero() -> None:
    assert robust_normalize([3.0, 3.0, 3.0]).tolist() == [0.0, 0.0, 0.0]
    assert robust_normalize([]).size == 0


def test_zscores_and_clamp() -> None:
    scores = zscores(np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]]), axis=0)
    assert scores[:, 0].tolist() == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-3)
    assert scores[:, 1].tolist() == [0.0, 0.0, 0.0]
    assert (clamp01(-1), clamp01(0.4), clamp01(7)) == (0.0, 0.4, 1.0)


# ----------------------------------------------------------------------- tags


def test_select_tags_picks_what_stands_out() -> None:
    rng = np.random.default_rng(1)
    similarities = (0.55 + rng.standard_normal((20, 3)) * 0.005).astype(np.float32)
    similarities[4, 1] = 0.75
    similarities[9, 0] = 0.72
    similarities[9, 2] = 0.70
    tags = select_tags(similarities, ["a", "b", "c"], max_tags=3, min_zscore=2.5)
    assert tags[4] == ("b",)
    assert set(tags[9]) == {"a", "c"}
    assert tags[0] == ()


def test_select_tags_with_few_segments_falls_back_to_ranking() -> None:
    similarities = np.array([[0.5, 0.7, 0.6]], dtype=np.float32)
    assert select_tags(similarities, ["a", "b", "c"], max_tags=3, min_zscore=1.0) == [("b", "c")]
    assert select_tags(
        similarities, ["a", "b", "c"], max_tags=3, min_zscore=1.0, min_similarity=0.65
    ) == [("b",)]


def test_footage_without_any_concept_gets_no_tags() -> None:
    """Standing out within the video is not enough: the match must be real."""
    rng = np.random.default_rng(3)
    similarities = (0.57 + rng.standard_normal((20, 4)) * 0.02).astype(np.float32)
    relative_only = select_tags(similarities, list("abcd"), max_tags=3, min_zscore=1.0)
    assert any(relative_only)
    with_floor = select_tags(
        similarities, list("abcd"), max_tags=3, min_zscore=1.0, min_similarity=0.65
    )
    assert all(tags == () for tags in with_floor)


def test_select_tags_edge_cases() -> None:
    empty = np.zeros((0, 3), dtype=np.float32)
    assert select_tags(empty, ["a", "b", "c"], max_tags=3, min_zscore=1.0) == []
    some = np.ones((2, 3), dtype=np.float32)
    assert select_tags(some, ["a", "b", "c"], max_tags=0, min_zscore=1.0) == [(), ()]


def test_reaction_presence_is_flat_without_a_clear_signal() -> None:
    flat = np.full((30, 2), 0.58, dtype=np.float32)
    assert reaction_presence(flat).max() == 0.0
    spiky = flat.copy()
    spiky[7, 0] = 0.80
    spiky[8, 1] = 0.69
    result = reaction_presence(spiky, floor=0.62, ceiling=0.76)
    assert result[7] == 1.0 and result[0] == 0.0
    assert result[8] == pytest.approx(0.5, abs=1e-5)
    assert reaction_presence(np.zeros((5, 0), dtype=np.float32)).tolist() == [0.0] * 5


def test_scene_describer_tags_segments_by_similarity() -> None:
    backend = FakeEmbedder()
    config = TagConfig(
        concepts=tuple(TagDefinition(id=c, label=c, prompt=f"a {c}") for c in CONCEPTS),
        reaction_cues=(),
        min_zscore=2.0,
    )
    describer = SceneDescriber(backend, config)
    vectors = np.zeros((12, backend.dimension), dtype=np.float32)
    vectors[:, 100] = 1.0  # unrelated baseline direction
    vectors[5] = backend.embed_queries(["explosion"])[0]
    descriptions = describer.describe(vectors)
    assert descriptions[5].tags == ("explosion",)
    assert descriptions[0].tags == ()
    assert all(item.reaction == 0.0 for item in descriptions)
    assert describer.describe(np.zeros((0, backend.dimension), dtype=np.float32)) == []


# ---------------------------------------------------------------------- score


def test_score_is_the_weighted_sum_from_the_spec() -> None:
    signals = ScoreSignals(
        semantic_similarity=0.9,
        audio_intensity=0.8,
        visual_motion=0.5,
        reaction_presence=0.2,
        novelty=1.0,
        duration_fit=0.6,
    )
    breakdown = compute_score(signals, ScoreWeights())
    expected = 0.35 * 0.9 + 0.20 * 0.8 + 0.15 * 0.5 + 0.15 * 0.2 + 0.10 * 1.0 + 0.05 * 0.6
    assert breakdown.total == pytest.approx(expected)
    assert breakdown.contributions["semantic_similarity"] == pytest.approx(0.315)
    assert sum(breakdown.contributions.values()) == pytest.approx(breakdown.total)


def test_score_bounds_and_clamping() -> None:
    weights = ScoreWeights()
    assert compute_score(ScoreSignals(1, 1, 1, 1, 1, 1), weights).total == pytest.approx(1.0)
    assert compute_score(ScoreSignals(0, 0, 0, 0, 0, 0), weights).total == 0.0
    wild = compute_score(ScoreSignals(5, -3, 0.5, 0.5, 0.5, 0.5), weights)
    assert wild.signals.semantic_similarity == 1.0 and wild.signals.audio_intensity == 0.0


def test_semantic_similarity_dominates_the_ranking() -> None:
    weights = ScoreWeights()
    relevant = compute_score(ScoreSignals(1.0, 0.2, 0.2, 0.0, 1.0, 1.0), weights)
    loud_but_off_topic = compute_score(ScoreSignals(0.2, 1.0, 0.2, 0.0, 1.0, 1.0), weights)
    assert relevant.total > loud_but_off_topic.total


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(15, 1.0), (30, 1.0), (45, 1.0), (8, 0.5), (11.5, 0.75), (52.5, 0.5), (60, 0.0), (4, 0.25)],
)
def test_duration_fit(seconds: float, expected: float) -> None:
    value = duration_fit(int(seconds * 1000), (15_000, 45_000), 8_000, 60_000)
    assert value == pytest.approx(expected)


def test_duration_fit_degenerate_input() -> None:
    assert duration_fit(0, (15_000, 45_000), 8_000, 60_000) == 0.0
    assert duration_fit(90_000, (15_000, 45_000), 8_000, 60_000) == 0.0


def test_temporal_overlap() -> None:
    assert temporal_overlap(0, 10, 20, 30) == 0.0
    assert temporal_overlap(0, 10, 5, 25) == pytest.approx(0.5)
    assert temporal_overlap(0, 100, 20, 30) == 1.0


def test_novelty_drops_for_repeated_content() -> None:
    vector = np.zeros(8, dtype=np.float32)
    vector[0] = 1.0
    other = np.zeros(8, dtype=np.float32)
    other[1] = 1.0
    assert novelty(0, 10_000, vector, [], []) == 1.0
    assert novelty(0, 10_000, vector, [], [other]) == 1.0  # different content
    assert novelty(0, 10_000, vector, [], [vector]) == 0.0  # same content, other video
    assert novelty(0, 10_000, vector, [(5_000, 15_000)], []) == pytest.approx(0.5)
    assert novelty(0, 10_000, None, [(0, 10_000)], []) == 0.0
    assert novelty(0, 10_000, vector, [], [np.zeros(8, dtype=np.float32)]) == 1.0
