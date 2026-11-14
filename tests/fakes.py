"""Deterministic stand-in for EmbeddingGemma 2 (tests only, never shipped as a backend).

Every "concept" owns one axis of the vector space. A text query lights up the axes
of the concept words it contains; a video window lights up the axis assigned to it
by ``window_concepts`` (keyed by window index, i.e. start time // stride). That
makes search results fully predictable without downloading any model.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from clipping_machine.embeddings.base import Matrix, WindowInput, l2_normalize

CONCEPTS = ("explosion", "chase", "drift", "cat", "menu")


class FakeEmbedder:
    def __init__(self, dimension: int = 768, window_concepts: dict[int, str] | None = None) -> None:
        self._dimension = dimension
        self.window_concepts = window_concepts or {}
        self.window_calls: list[WindowInput] = []
        self._next_window = 0

    @property
    def model_name(self) -> str:
        return "fake/embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_loaded(self) -> bool:
        return True

    def _noise(self, seed: str) -> NDArray[np.float32]:
        digest = hashlib.sha256(seed.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return (rng.standard_normal(self._dimension) * 0.01).astype(np.float32)

    def _concept_vector(self, concepts: Sequence[str], seed: str) -> NDArray[np.float32]:
        vector = self._noise(seed)
        for concept in concepts:
            vector[CONCEPTS.index(concept)] += 1.0
        return vector

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        rows = [
            self._concept_vector([c for c in CONCEPTS if c in text.lower()], f"q:{text}")
            for text in texts
        ]
        return l2_normalize(np.stack(rows)) if rows else np.zeros((0, self._dimension), np.float32)

    def embed_windows(self, windows: Sequence[WindowInput]) -> Matrix:
        rows = []
        for window in windows:
            index = self._next_window
            self._next_window += 1
            self.window_calls.append(window)
            concept = self.window_concepts.get(index)
            rows.append(self._concept_vector([concept] if concept else [], f"w:{index}"))
        return l2_normalize(np.stack(rows)) if rows else np.zeros((0, self._dimension), np.float32)

    def embed_images(self, images: Sequence[Path]) -> Matrix:
        rows = [self._noise(f"i:{path}") for path in images]
        return l2_normalize(np.stack(rows)) if rows else np.zeros((0, self._dimension), np.float32)

    def embed_audio(self, clips: Sequence[NDArray[np.float32]], sample_rate: int) -> Matrix:
        rows = [self._noise(f"a:{index}:{sample_rate}") for index, _ in enumerate(clips)]
        return l2_normalize(np.stack(rows)) if rows else np.zeros((0, self._dimension), np.float32)

    def reset(self) -> None:
        self._next_window = 0
        self.window_calls.clear()
