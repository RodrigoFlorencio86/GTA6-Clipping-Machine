"""EmbeddingGemma client: payload shapes and prompts, checked against a stub model.

The real model is exercised only by ``pytest -m model`` (see the last test).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from PIL import Image

from clipping_machine.config import EmbeddingConfig
from clipping_machine.embeddings.base import WindowInput, l2_normalize, truncate
from clipping_machine.embeddings.embeddinggemma_client import (
    EmbeddingGemmaClient,
    EmbeddingModelError,
    prepare_audio,
    resolve_vision_tokens,
)


class StubModel:
    """Records what the client sends to ``SentenceTransformer.encode``."""

    prompts: ClassVar[dict[str, str]] = {"SearchQuery": "task: search result | query: "}
    instances: ClassVar[list[StubModel]] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.init_kwargs = kwargs
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []
        self.nan = False
        StubModel.instances.append(self)

    def encode(self, inputs: list[Any], **kwargs: Any) -> np.ndarray:
        self.calls.append((inputs, kwargs))
        dimension = kwargs.get("truncate_dim") or 768
        output = np.ones((len(inputs), dimension), dtype=np.float32)
        if self.nan:
            output[0, 0] = np.nan
        return output


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[StubModel]:
    StubModel.instances.clear()
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return StubModel


@pytest.fixture
def frames(tmp_path: Path) -> tuple[Path, ...]:
    paths = []
    for index, size in enumerate([(64, 36), (64, 36), (32, 18)]):
        path = tmp_path / f"kf_{index}.jpg"
        Image.new("RGB", size, (index * 80, 10, 10)).save(path)
        paths.append(path)
    return tuple(paths)


def test_model_is_loaded_lazily_with_safe_settings(stub: type[StubModel]) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu", local_files_only=True))
    assert not client.is_loaded and stub.instances == []
    client.load()
    client.load()
    assert len(stub.instances) == 1
    model = stub.instances[0]
    assert model.name == "google/embeddinggemma-2"
    assert model.init_kwargs["device"] == "cpu"
    assert model.init_kwargs["local_files_only"] is True
    # float16 produces NaNs with this model; CPU must use float32.
    assert str(model.init_kwargs["model_kwargs"]["dtype"]) == "torch.float32"
    assert model.init_kwargs["config_kwargs"] is None
    assert client.device == "cpu" and client.vision_tokens == 70


def test_audio_encoder_can_be_skipped(stub: type[StubModel]) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu", include_audio=False))
    client.load()
    assert stub.instances[0].init_kwargs["config_kwargs"] == {"audio_config": None}


def test_queries_use_the_search_prompt(stub: type[StubModel]) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu"))
    vectors = client.embed_queries(["police chase", "explosão"])
    inputs, kwargs = stub.instances[0].calls[0]
    assert inputs == ["police chase", "explosão"]
    assert kwargs["prompt_name"] == "SearchQuery"
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["truncate_dim"] is None
    assert vectors.shape == (2, 768)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert client.embed_queries([]).shape == (0, 768)


def test_window_payload_is_frames_plus_audio(
    stub: type[StubModel], frames: tuple[Path, ...]
) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu"))
    audio = np.linspace(-0.5, 0.5, 8 * 16000, dtype=np.float32)
    client.embed_windows([WindowInput(frames=frames, audio=audio)])
    inputs, kwargs = stub.instances[0].calls[0]
    payload = inputs[0]

    # Video first, then audio: the model sees the frames followed by the sound.
    assert list(payload) == ["video", "audio"]
    assert all(isinstance(frame, Image.Image) and frame.mode == "RGB" for frame in payload["video"])
    assert {frame.size for frame in payload["video"]} == {(64, 36)}  # resized to one size
    assert payload["audio"]["sampling_rate"] == 16000
    assert payload["audio"]["array"].ndim == 1 and payload["audio"]["array"].dtype == np.float32
    # No task prompt for media, and frames are used exactly as extracted.
    assert "prompt_name" not in kwargs
    assert kwargs["processing_kwargs"] == {
        "video": {"do_sample_frames": False, "add_timestamps": False, "max_soft_tokens": 70}
    }
    assert kwargs["batch_size"] == 4


def test_silent_or_missing_audio_is_left_out(
    stub: type[StubModel], frames: tuple[Path, ...]
) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu"))
    client.embed_windows(
        [
            WindowInput(frames=frames[:2]),
            WindowInput(frames=frames[:2], audio=np.zeros(16000, dtype=np.float32)),
        ]
    )
    inputs, _ = stub.instances[0].calls[0]
    assert all(list(payload) == ["video"] for payload in inputs)


def test_audio_disabled_in_config_is_never_sent(
    stub: type[StubModel], frames: tuple[Path, ...]
) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu", include_audio=False))
    client.embed_windows([WindowInput(frames=frames[:1], audio=np.ones(16000, dtype=np.float32))])
    assert list(stub.instances[0].calls[0][0][0]) == ["video"]


def test_window_without_frames_is_an_error(stub: type[StubModel]) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu"))
    with pytest.raises(ValueError):
        client.embed_windows([WindowInput(frames=())])


def test_matryoshka_dimension_is_requested_and_renormalized(stub: type[StubModel]) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu", dimension=256))
    vectors = client.embed_queries(["drift"])
    assert stub.instances[0].calls[0][1]["truncate_dim"] == 256
    assert vectors.shape == (1, 256)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0)


def test_non_finite_output_is_rejected(stub: type[StubModel]) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu"))
    client.load()
    stub.instances[0].nan = True
    with pytest.raises(EmbeddingModelError, match="non-finite"):
        client.embed_queries(["anything"])


def test_load_failure_explains_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("not found in cache")

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu", local_files_only=True))
    with pytest.raises(EmbeddingModelError, match="download-model"):
        client.load()


def test_images_and_audio_queries(stub: type[StubModel], frames: tuple[Path, ...]) -> None:
    client = EmbeddingGemmaClient(EmbeddingConfig(device="cpu"))
    assert client.embed_images(frames[:2]).shape == (2, 768)
    tone = np.sin(np.arange(16000) / 10).astype(np.float32)
    assert client.embed_audio([tone], 16000).shape == (1, 768)
    with pytest.raises(ValueError):
        client.embed_audio([np.zeros(16000, dtype=np.float32)], 16000)
    assert client.embed_images([]).shape == (0, 768)


def test_prepare_audio_rules() -> None:
    tone = np.sin(np.arange(40 * 16000) / 10).astype(np.float32)
    prepared = prepare_audio(tone, 16000)
    assert prepared is not None and prepared.size == 30 * 16000  # capped at 30 s
    assert prepare_audio(None, 16000) is None
    assert prepare_audio(tone[:100], 16000) is None  # too short to mean anything
    assert prepare_audio(np.zeros(16000, dtype=np.float32), 16000) is None
    with pytest.raises(ValueError, match="16000"):
        prepare_audio(tone, 44100)  # the library would not resample: fail loudly


def test_vision_token_budget() -> None:
    assert resolve_vision_tokens("auto", "cpu") == 70
    assert resolve_vision_tokens("auto", "cuda") == 140
    assert resolve_vision_tokens(280, "cpu") == 280


def test_vector_helpers() -> None:
    matrix = np.array([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    normalized = l2_normalize(matrix)
    assert normalized[0].tolist() == pytest.approx([0.6, 0.8, 0.0, 0.0])
    assert normalized[1].tolist() == [0.0, 0.0, 0.0, 0.0]
    shortened = truncate(np.array([[3.0, 4.0, 12.0, 0.0]]), 2)
    assert shortened[0].tolist() == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError):
        truncate(matrix, 8)


@pytest.mark.model
def test_real_model_text_and_video_share_one_space(tmp_path: Path) -> None:
    """Opt-in (``pytest -m model``): loads the real EmbeddingGemma 2 weights."""
    red, blue = tmp_path / "red.jpg", tmp_path / "blue.jpg"
    Image.new("RGB", (320, 180), (220, 20, 20)).save(red)
    Image.new("RGB", (320, 180), (20, 20, 220)).save(blue)
    client = EmbeddingGemmaClient(EmbeddingConfig())
    windows = client.embed_windows(
        [WindowInput(frames=(red, red)), WindowInput(frames=(blue, blue))]
    )
    queries = client.embed_queries(["a solid red image", "a solid blue image"])
    assert windows.shape == (2, 768) and queries.shape == (2, 768)
    similarity = queries @ windows.T
    assert similarity[0, 0] > similarity[0, 1]
    assert similarity[1, 1] > similarity[1, 0]
