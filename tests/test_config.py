from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from clipping_machine.config import (
    EmbeddingConfig,
    ScoreWeights,
    SearchConfig,
    VideoConfig,
    load_settings,
)


def test_defaults_match_the_spec(tmp_path: Path) -> None:
    settings = load_settings(env={"GCM_DATA_DIR": str(tmp_path)})
    assert settings.video.window_duration_seconds == 8
    assert settings.video.window_overlap_seconds == 2
    assert settings.video.keyframe_interval_seconds == 2
    assert settings.video.audio_sample_rate == 16000
    assert settings.embedding.model == "google/embeddinggemma-2"
    assert settings.embedding.dimension == 768
    assert settings.search.min_similarity == 0.65
    assert settings.search.merge_gap_seconds == 3
    assert settings.search.target_clip_duration_seconds == (15, 45)
    assert settings.search.max_clip_duration_seconds == 60
    assert settings.search.min_clip_duration_seconds == 8
    assert settings.search.top_k == 50
    assert (settings.export.width, settings.export.height) == (1080, 1920)
    assert set(settings.upload.allowed_extensions) == {".mp4", ".mov", ".mkv", ".webm"}


def test_score_weights_match_the_spec_formula() -> None:
    weights = ScoreWeights()
    assert weights.model_dump() == {
        "semantic_similarity": 0.35,
        "audio_intensity": 0.20,
        "visual_motion": 0.15,
        "reaction_presence": 0.15,
        "novelty": 0.10,
        "duration_fit": 0.05,
    }


def test_score_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        ScoreWeights(semantic_similarity=0.9)


def test_overlap_must_be_shorter_than_window() -> None:
    with pytest.raises(ValidationError):
        VideoConfig(window_duration_seconds=4, window_overlap_seconds=4)


def test_embedding_dimension_must_be_a_matryoshka_size() -> None:
    assert EmbeddingConfig(dimension=256).dimension == 256
    with pytest.raises(ValidationError):
        EmbeddingConfig(dimension=300)


def test_vision_tokens_validation() -> None:
    assert EmbeddingConfig(vision_tokens=70).vision_tokens == 70
    with pytest.raises(ValidationError):
        EmbeddingConfig(vision_tokens=100)


def test_clip_durations_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        SearchConfig(target_clip_duration_seconds=(15, 90))


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "processing.yaml").write_text("search:\n  topk: 5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(config_dir=tmp_path, env={"GCM_DATA_DIR": str(tmp_path)})


def test_environment_overrides(tmp_path: Path) -> None:
    settings = load_settings(
        env={
            "GCM_DATA_DIR": str(tmp_path / "elsewhere"),
            "GCM_EMBEDDING_MODEL": "/models/embeddinggemma-2",
            "GCM_DEVICE": "cpu",
            "GCM_OFFLINE": "1",
        }
    )
    assert settings.data_dir == tmp_path / "elsewhere"
    assert settings.embedding.model == "/models/embeddinggemma-2"
    assert settings.embedding.device == "cpu"
    assert settings.embedding.local_files_only is True


def test_tags_are_loaded_from_yaml(tmp_path: Path) -> None:
    settings = load_settings(env={"GCM_DATA_DIR": str(tmp_path)})
    ids = [concept.id for concept in settings.tags.concepts]
    assert "police_chase" in ids and "explosion" in ids
    assert len(ids) == len(set(ids))
    assert settings.tags.reaction_cues
