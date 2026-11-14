"""EmbeddingGemma 2 backend (``google/embeddinggemma-2`` via sentence-transformers).

The model maps text, images, video frames and audio into one shared vector space.
It is used here strictly as an embedding model: no generation, no transcription,
no editing decisions.

Input conventions that matter (verified against sentence-transformers 6.3 /
transformers 5.23, see docs/MODELS.md):

* Video is passed as already-decoded RGB frames of identical size, so no video
  decoding library is required and nothing but the sampled frames leaves FFmpeg.
* Audio is a 1-D float32 numpy array, mono, 16 kHz. The library does not resample.
* Text queries use the ``SearchQuery`` task prompt; media inputs take no prompt.
* float16 is never used: the model card documents NaNs in that precision.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from clipping_machine.config import EmbeddingConfig
from clipping_machine.embeddings.base import Matrix, WindowInput, l2_normalize

MODEL_SAMPLE_RATE = 16000
# The model card caps a single audio input at 30 s; windows are far shorter.
_MAX_AUDIO_SAMPLES = 30 * MODEL_SAMPLE_RATE
# Shorter clips than this carry no usable signal for the audio encoder.
_MIN_AUDIO_SAMPLES = MODEL_SAMPLE_RATE // 10
_VIDEO_KWARGS: dict[str, Any] = {"do_sample_frames": False, "add_timestamps": False}
_CPU_VISION_TOKENS = 70
_GPU_VISION_TOKENS = 140


class EmbeddingModelError(RuntimeError):
    """The embedding model could not be loaded or produced invalid output."""


def resolve_device(requested: str) -> str:
    """``auto`` → cuda, then mps, then cpu."""
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> Any:
    """bfloat16 where the hardware supports it natively, float32 elsewhere."""
    import torch

    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def resolve_vision_tokens(requested: str | int, device: str) -> int:
    """Soft tokens per frame: fewer on CPU, where the vision encoder dominates the cost."""
    if requested != "auto":
        return int(requested)
    return _CPU_VISION_TOKENS if device == "cpu" else _GPU_VISION_TOKENS


def prepare_audio(
    samples: NDArray[np.float32] | None, sample_rate: int
) -> NDArray[np.float32] | None:
    """Validate an audio chunk for the model; ``None`` means "embed frames only"."""
    if samples is None:
        return None
    if sample_rate != MODEL_SAMPLE_RATE:
        raise ValueError(f"audio must be sampled at {MODEL_SAMPLE_RATE} Hz, got {sample_rate} Hz")
    audio = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
    if audio.size < _MIN_AUDIO_SAMPLES or not np.any(audio):
        return None
    return audio[:_MAX_AUDIO_SAMPLES]


class EmbeddingGemmaClient:
    """Lazy, thread-safe wrapper around the sentence-transformers model."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model: Any = None
        self._lock = threading.RLock()
        self.device: str | None = None
        self.vision_tokens: int | None = None

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the weights (downloads them on first use unless offline)."""
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                device = resolve_device(self._config.device)
                config_kwargs: dict[str, Any] = {}
                if not self._config.include_audio:
                    # Skips the 300M-parameter audio encoder entirely.
                    config_kwargs["audio_config"] = None
                model = SentenceTransformer(
                    self._config.model,
                    device=device,
                    local_files_only=self._config.local_files_only,
                    model_kwargs={"dtype": resolve_dtype(self._config.dtype, device)},
                    config_kwargs=config_kwargs or None,
                )
            except Exception as exc:
                hint = (
                    " The model is not in the local cache and offline mode is on; run "
                    "`clipping-machine download-model` once with network access."
                    if self._config.local_files_only
                    else ""
                )
                raise EmbeddingModelError(
                    f"could not load embedding model '{self._config.model}': {exc}.{hint}"
                ) from exc
            if self._config.query_prompt_name not in model.prompts:
                raise EmbeddingModelError(
                    f"model has no prompt named '{self._config.query_prompt_name}'; "
                    f"available: {sorted(model.prompts)}"
                )
            self._model = model
            self.device = device
            self.vision_tokens = resolve_vision_tokens(self._config.vision_tokens, device)

    # ------------------------------------------------------------------ public

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode(list(texts), prompt_name=self._config.query_prompt_name)

    def embed_windows(self, windows: Sequence[WindowInput]) -> Matrix:
        if not windows:
            return np.zeros((0, self.dimension), dtype=np.float32)
        self.load()
        inputs = [self._window_payload(window) for window in windows]
        video_kwargs = {**_VIDEO_KWARGS, "max_soft_tokens": self.vision_tokens}
        return self._encode(inputs, processing_kwargs={"video": video_kwargs})

    def embed_images(self, images: Sequence[Path]) -> Matrix:
        if not images:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode([self._load_frame(path) for path in images])

    def embed_audio(self, clips: Sequence[NDArray[np.float32]], sample_rate: int) -> Matrix:
        if not clips:
            return np.zeros((0, self.dimension), dtype=np.float32)
        prepared = [prepare_audio(clip, sample_rate) for clip in clips]
        if any(clip is None for clip in prepared):
            raise ValueError("audio clips must be non-silent and at least 0.1 s long")
        return self._encode(
            [{"array": clip, "sampling_rate": MODEL_SAMPLE_RATE} for clip in prepared]
        )

    # ---------------------------------------------------------------- internals

    def _window_payload(self, window: WindowInput) -> dict[str, Any]:
        if not window.frames:
            raise ValueError("a window needs at least one frame")
        frames = [self._load_frame(path) for path in window.frames]
        size = frames[0].size
        frames = [frame if frame.size == size else frame.resize(size) for frame in frames]
        payload: dict[str, Any] = {"video": frames}
        audio = (
            prepare_audio(window.audio, window.sample_rate) if self._config.include_audio else None
        )
        if audio is not None:
            payload["audio"] = {"array": audio, "sampling_rate": MODEL_SAMPLE_RATE}
        return payload

    @staticmethod
    def _load_frame(path: Path) -> Any:
        from PIL import Image

        with Image.open(path) as image:
            return image.convert("RGB")

    def _encode(self, inputs: list[Any], **kwargs: Any) -> Matrix:
        self.load()
        truncate_dim = None if self.dimension == 768 else self.dimension
        with self._lock:
            output = self._model.encode(
                inputs,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize,
                truncate_dim=truncate_dim,
                convert_to_numpy=True,
                show_progress_bar=False,
                **kwargs,
            )
        matrix = np.asarray(output, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape != (len(inputs), self.dimension):
            raise EmbeddingModelError(
                f"model returned shape {matrix.shape}, expected {(len(inputs), self.dimension)}"
            )
        if not np.all(np.isfinite(matrix)):
            raise EmbeddingModelError("model returned non-finite embeddings (NaN/Inf)")
        return l2_normalize(matrix) if self._config.normalize else matrix
