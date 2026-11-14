"""Zero-shot tagging of segments with the embedding model.

There is no captioning model here. Each concept in ``configs/tags.yaml`` is embedded
as a text query once; a segment gets a concept only when its similarity both stands
out for that concept *within the same video* and clears an absolute floor, so footage
that contains none of the concepts gets no tags at all. The tags become the
human-readable "reason" shown next to a moment. They are retrieval hints, not ground
truth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from clipping_machine.analysis.normalization import zscores
from clipping_machine.config import TagConfig
from clipping_machine.embeddings.base import EmbeddingBackend, Matrix

# With very few segments a z-score is meaningless; fall back to plain ranking.
_MIN_SEGMENTS_FOR_ZSCORE = 8


@dataclass(frozen=True)
class SegmentDescription:
    tags: tuple[str, ...]
    reaction: float


class SceneDescriber:
    def __init__(self, backend: EmbeddingBackend, config: TagConfig) -> None:
        self._backend = backend
        self._config = config
        self._concept_vectors: Matrix | None = None
        self._reaction_vectors: Matrix | None = None

    def _concepts(self) -> Matrix:
        if self._concept_vectors is None:
            prompts = [concept.prompt for concept in self._config.concepts]
            self._concept_vectors = (
                self._backend.embed_queries(prompts)
                if prompts
                else np.zeros((0, self._backend.dimension), dtype=np.float32)
            )
        return self._concept_vectors

    def _reactions(self) -> Matrix:
        if self._reaction_vectors is None:
            cues = list(self._config.reaction_cues)
            self._reaction_vectors = (
                self._backend.embed_queries(cues)
                if cues
                else np.zeros((0, self._backend.dimension), dtype=np.float32)
            )
        return self._reaction_vectors

    def describe(self, segment_vectors: Matrix) -> list[SegmentDescription]:
        """Tags and reaction-presence for every segment of one video."""
        count = segment_vectors.shape[0]
        if count == 0:
            return []
        tags = select_tags(
            segment_vectors @ self._concepts().T,
            [concept.id for concept in self._config.concepts],
            max_tags=self._config.max_tags_per_segment,
            min_zscore=self._config.min_zscore,
            min_similarity=self._config.min_similarity,
            max_rank=self._config.max_rank or None,
        )
        reaction = reaction_presence(
            segment_vectors @ self._reactions().T,
            floor=self._config.reaction_floor,
            ceiling=self._config.reaction_ceiling,
        )
        return [SegmentDescription(tags[i], float(reaction[i])) for i in range(count)]


def select_tags(
    similarities: NDArray[np.float32],
    concept_ids: Sequence[str],
    *,
    max_tags: int,
    min_zscore: float,
    min_similarity: float = 0.0,
    max_rank: int | None = None,
) -> list[tuple[str, ...]]:
    """Pick, per segment, the concepts that stand out and are similar enough.

    ``similarities`` has shape ``(segments, concepts)`` and holds cosine values. With
    ``max_rank``, a concept must also be among the segment's ``max_rank`` most similar
    concepts. Without that, a concept that is rare in the video (low baseline) wins a high
    z-score from any small bump, which tagged a dark office "parachute" on real footage.
    """
    segments, concepts = similarities.shape if similarities.ndim == 2 else (0, 0)
    if segments == 0 or concepts == 0 or max_tags == 0:
        return [() for _ in range(max(segments, 0))]

    if segments < _MIN_SEGMENTS_FOR_ZSCORE:
        limit = min(max_tags, 2, concepts)
        order = np.argsort(-similarities, axis=1)[:, :limit]
        return [
            tuple(concept_ids[j] for j in row if similarities[i, j] >= min_similarity)
            for i, row in enumerate(order)
        ]

    scores = zscores(similarities, axis=0)
    rank_limit = concepts if max_rank is None else max(1, max_rank)
    result: list[tuple[str, ...]] = []
    for i, row in enumerate(scores):
        closest = {int(j) for j in np.argsort(-similarities[i])[:rank_limit]}
        ranked = [
            int(j)
            for j in np.argsort(-row)
            if row[j] >= min_zscore and similarities[i, j] >= min_similarity and int(j) in closest
        ]
        result.append(tuple(concept_ids[j] for j in ranked[:max_tags]))
    return result


def reaction_presence(
    similarities: NDArray[np.float32], *, floor: float = 0.62, ceiling: float = 0.76
) -> NDArray[np.float32]:
    """Heuristic 0–1 signal: how much a segment resembles "someone reacting".

    Takes the best-matching reaction cue per segment and maps its cosine similarity
    from ``[floor, ceiling]`` to ``[0, 1]``. The scale is absolute, so a recording
    with no commentary or webcam stays near zero instead of producing relative peaks.
    """
    if similarities.ndim != 2 or similarities.shape[0] == 0 or similarities.shape[1] == 0:
        return np.zeros(similarities.shape[0] if similarities.ndim == 2 else 0, dtype=np.float32)
    best = np.asarray(np.max(similarities, axis=1), dtype=np.float64)
    scaled: NDArray[np.float32] = np.asarray(
        np.clip((best - floor) / (ceiling - floor), 0.0, 1.0), dtype=np.float32
    )
    return scaled
