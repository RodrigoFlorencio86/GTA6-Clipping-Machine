"""Local vector index backed by LanceDB (embedded, file-based, no server)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


@dataclass(frozen=True)
class VectorRecord:
    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    vector: NDArray[np.float32]


@dataclass(frozen=True)
class VectorHit:
    segment_id: str
    video_id: str
    start_ms: int
    end_ms: int
    similarity: float


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _check_video_id(video_id: str) -> str:
    if not _VIDEO_ID.match(video_id):
        raise ValueError(f"invalid video id: {video_id!r}")
    return video_id


class VectorStore:
    """Cosine-similarity index over segment embeddings.

    One table per embedding dimension; rows also carry the model name so vectors
    from different models are never compared.
    """

    def __init__(self, index_dir: Path, dimension: int, model_name: str) -> None:
        import lancedb
        import pyarrow as pa

        self.dimension = dimension
        self.model_name = model_name
        index_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(index_dir))
        self._schema = pa.schema(
            [
                pa.field("segment_id", pa.string()),
                pa.field("video_id", pa.string()),
                pa.field("start_ms", pa.int64()),
                pa.field("end_ms", pa.int64()),
                pa.field("embedding_model", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dimension)),
            ]
        )
        self._table = self._db.create_table(
            f"segments_d{dimension}", schema=self._schema, exist_ok=True
        )

    def _model_filter(self) -> str:
        return f"embedding_model = {_quote(self.model_name)}"

    def add(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        rows: list[dict[str, Any]] = []
        for record in records:
            vector = np.asarray(record.vector, dtype=np.float32)
            if vector.shape != (self.dimension,):
                raise ValueError(
                    f"expected a vector of dimension {self.dimension}, got shape {vector.shape}"
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"segment {record.segment_id} has a non-finite embedding")
            rows.append(
                {
                    "segment_id": record.segment_id,
                    "video_id": _check_video_id(record.video_id),
                    "start_ms": int(record.start_ms),
                    "end_ms": int(record.end_ms),
                    "embedding_model": self.model_name,
                    "vector": vector.tolist(),
                }
            )
        self._table.add(rows)

    def search(
        self, query: NDArray[np.float32], *, top_k: int, video_id: str | None = None
    ) -> list[VectorHit]:
        """Nearest segments by cosine similarity, best first."""
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"expected a query of dimension {self.dimension}, got shape {vector.shape}"
            )
        where = self._model_filter()
        if video_id is not None:
            where += f" AND video_id = {_quote(_check_video_id(video_id))}"
        rows = (
            self._table.search(vector.tolist())
            .distance_type("cosine")
            .where(where, prefilter=True)
            .limit(top_k)
            .to_list()
        )
        hits = [
            VectorHit(
                segment_id=row["segment_id"],
                video_id=row["video_id"],
                start_ms=int(row["start_ms"]),
                end_ms=int(row["end_ms"]),
                similarity=1.0 - float(row["_distance"]),
            )
            for row in rows
        ]
        return sorted(hits, key=lambda hit: hit.similarity, reverse=True)

    def vectors_between(self, video_id: str, start_ms: int, end_ms: int) -> NDArray[np.float32]:
        """Embeddings of the segments overlapping ``[start_ms, end_ms)`` (may be empty)."""
        where = (
            f"{self._model_filter()} AND video_id = {_quote(_check_video_id(video_id))} "
            f"AND start_ms < {int(end_ms)} AND end_ms > {int(start_ms)}"
        )
        rows = self._table.search().where(where).select(["vector"]).limit(100_000).to_list()
        if not rows:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.asarray([row["vector"] for row in rows], dtype=np.float32)

    def count(self, video_id: str | None = None) -> int:
        if video_id is None:
            return int(self._table.count_rows())
        return int(self._table.count_rows(f"video_id = {_quote(_check_video_id(video_id))}"))

    def delete_video(self, video_id: str) -> None:
        self._table.delete(f"video_id = {_quote(_check_video_id(video_id))}")
