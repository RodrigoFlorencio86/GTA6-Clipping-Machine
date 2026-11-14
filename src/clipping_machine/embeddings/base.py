"""Backend-agnostic embedding interface.

The application code depends on this protocol, never on a concrete model, so the
pipeline can be unit-tested with a deterministic fake and no model download.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Vector = NDArray[np.float32]
Matrix = NDArray[np.float32]


@dataclass(frozen=True)
class WindowInput:
    """Everything the model sees for one temporal window of the video."""

    frames: tuple[Path, ...]
    audio: NDArray[np.float32] | None = None
    sample_rate: int = 16000


@runtime_checkable
class EmbeddingBackend(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        """Text queries → unit vectors, shape ``(len(texts), dimension)``."""
        ...

    def embed_windows(self, windows: Sequence[WindowInput]) -> Matrix:
        """Video windows (frames + optional audio) → unit vectors."""
        ...

    def embed_images(self, images: Sequence[Path]) -> Matrix:
        """Still images → unit vectors (reference-image search)."""
        ...

    def embed_audio(self, clips: Sequence[NDArray[np.float32]], sample_rate: int) -> Matrix:
        """Mono audio clips → unit vectors (reference-audio search)."""
        ...


def l2_normalize(matrix: NDArray[np.floating]) -> Matrix:
    """Row-wise L2 normalization; zero rows stay zero."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return (array / np.where(norms == 0.0, 1.0, norms)).astype(np.float32)


def truncate(matrix: NDArray[np.floating], dimension: int) -> Matrix:
    """Matryoshka truncation: keep the leading dimensions, then re-normalize."""
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if dimension > array.shape[1]:
        raise ValueError(f"cannot truncate {array.shape[1]}-d vectors to {dimension}-d")
    return l2_normalize(array[:, :dimension])
