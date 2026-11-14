"""Text query → embedding → nearest segments."""

from __future__ import annotations

from clipping_machine.analysis.normalization import clamp01
from clipping_machine.config import CalibrationConfig
from clipping_machine.embeddings.base import EmbeddingBackend
from clipping_machine.search.interval_merger import Hit
from clipping_machine.search.vector_store import VectorHit, VectorStore


def calibrate(raw_similarity: float, config: CalibrationConfig) -> float:
    """Spread cosine similarity over 0–1 for use as a score component.

    Text↔video cosine occupies a narrow band (unrelated ≈ ``floor``, clear match ≈
    ``ceiling``); the linear map makes the semantic term of the editorial score
    discriminative. Thresholding always uses the raw cosine, never this value.
    """
    if not config.enabled:
        return clamp01(raw_similarity)
    return clamp01((raw_similarity - config.floor) / (config.ceiling - config.floor))


class QueryEngine:
    def __init__(
        self, backend: EmbeddingBackend, store: VectorStore, calibration: CalibrationConfig
    ) -> None:
        self._backend = backend
        self._store = store
        self._calibration = calibration

    def warm_up(self) -> None:
        """Load the model if the backend loads lazily (kept out of latency figures)."""
        load = getattr(self._backend, "load", None)
        if callable(load):
            load()

    def search(self, query: str, *, top_k: int, video_id: str | None = None) -> list[VectorHit]:
        """Raw nearest-neighbour hits for a natural-language query, best first."""
        text = query.strip()
        if not text:
            raise ValueError("the search query is empty")
        vector = self._backend.embed_queries([text])[0]
        return self._store.search(vector, top_k=top_k, video_id=video_id)

    @staticmethod
    def to_hit(hit: VectorHit) -> Hit:
        return Hit(hit.segment_id, hit.start_ms, hit.end_ms, hit.similarity)

    def score_component(self, cosine: float) -> float:
        """Cosine similarity rescaled for the editorial score."""
        return calibrate(cosine, self._calibration)
