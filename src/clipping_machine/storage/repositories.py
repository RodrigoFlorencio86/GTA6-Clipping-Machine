"""Repositories: the only place that knows the SQL."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence

from clipping_machine.storage.database import Database, utc_now
from clipping_machine.storage.models import (
    Clip,
    ClipEvent,
    ClipStatus,
    Scene,
    Segment,
    Video,
    VideoStatus,
)


def _video(row: sqlite3.Row) -> Video:
    return Video(
        video_id=row["video_id"],
        original_name=row["original_name"],
        path=row["path"],
        managed_copy=bool(row["managed_copy"]),
        size_bytes=row["size_bytes"],
        fingerprint=row["fingerprint"],
        duration_ms=row["duration_ms"],
        width=row["width"],
        height=row["height"],
        fps=row["fps"],
        video_codec=row["video_codec"],
        audio_codec=row["audio_codec"],
        rights_confirmed_at=row["rights_confirmed_at"],
        created_at=row["created_at"],
        status=VideoStatus(row["status"]),
        stage=row["stage"],
        progress=row["progress"],
        error=row["error"],
        segments_count=row["segments_count"],
        indexed_at=row["indexed_at"],
        index_seconds=row["index_seconds"],
        embedding_model=row["embedding_model"],
    )


def _segment(row: sqlite3.Row) -> Segment:
    return Segment(
        segment_id=row["segment_id"],
        video_id=row["video_id"],
        seg_index=row["seg_index"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        preview_frame_path=row["preview_frame_path"],
        embedding_model=row["embedding_model"],
        embedding_dimension=row["embedding_dimension"],
        audio_energy=row["audio_energy"],
        audio_db=row["audio_db"],
        motion=row["motion"],
        motion_raw=row["motion_raw"],
        scene_cuts=row["scene_cuts"],
        reaction=row["reaction"],
        tags=list(json.loads(row["tags"])),
        transcript=row["transcript"],
        status=row["status"],
    )


def _clip(row: sqlite3.Row) -> Clip:
    return Clip(
        clip_id=row["clip_id"],
        video_id=row["video_id"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        query=row["query"],
        title=row["title"],
        score=row["score"],
        score_breakdown=dict(json.loads(row["score_breakdown"])),
        reason=row["reason"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        status=ClipStatus(row["status"]),
        draft_path=row["draft_path"],
        output_path=row["output_path"],
        thumbnail_path=row["thumbnail_path"],
        review_note=row["review_note"],
        exported_at=row["exported_at"],
    )


class VideoRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, video: Video) -> None:
        with self._db.connect() as connection:
            connection.execute(
                """
                INSERT INTO videos (
                    video_id, original_name, path, managed_copy, size_bytes, fingerprint,
                    duration_ms, width, height, fps, video_codec, audio_codec,
                    rights_confirmed_at, created_at, status, stage, progress, error,
                    segments_count, indexed_at, index_seconds, embedding_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video.video_id,
                    video.original_name,
                    video.path,
                    int(video.managed_copy),
                    video.size_bytes,
                    video.fingerprint,
                    video.duration_ms,
                    video.width,
                    video.height,
                    video.fps,
                    video.video_codec,
                    video.audio_codec,
                    video.rights_confirmed_at,
                    video.created_at,
                    video.status.value,
                    video.stage,
                    video.progress,
                    video.error,
                    video.segments_count,
                    video.indexed_at,
                    video.index_seconds,
                    video.embedding_model,
                ),
            )

    def get(self, video_id: str) -> Video | None:
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
        return _video(row) if row else None

    def require(self, video_id: str) -> Video:
        video = self.get(video_id)
        if video is None:
            raise KeyError(f"video not found: {video_id}")
        return video

    def find_by_fingerprint(self, fingerprint: str) -> Video | None:
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM videos WHERE fingerprint = ? LIMIT 1", (fingerprint,)
            ).fetchone()
        return _video(row) if row else None

    def all(self, status: VideoStatus | None = None) -> list[Video]:
        query = "SELECT * FROM videos"
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status.value,)
        query += " ORDER BY created_at DESC, video_id"
        with self._db.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_video(row) for row in rows]

    def set_progress(self, video_id: str, stage: str, progress: float) -> None:
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE videos SET stage = ?, progress = ? WHERE video_id = ?",
                (stage, max(0.0, min(1.0, progress)), video_id),
            )

    def mark_indexing(self, video_id: str) -> None:
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE videos SET status = ?, stage = '', progress = 0, error = NULL "
                "WHERE video_id = ?",
                (VideoStatus.INDEXING.value, video_id),
            )

    def mark_indexed(
        self, video_id: str, *, segments_count: int, index_seconds: float, embedding_model: str
    ) -> None:
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE videos SET status = ?, stage = 'done', progress = 1, error = NULL, "
                "segments_count = ?, indexed_at = ?, index_seconds = ?, embedding_model = ? "
                "WHERE video_id = ?",
                (
                    VideoStatus.INDEXED.value,
                    segments_count,
                    utc_now(),
                    index_seconds,
                    embedding_model,
                    video_id,
                ),
            )

    def mark_failed(self, video_id: str, error: str) -> None:
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE videos SET status = ?, error = ? WHERE video_id = ?",
                (VideoStatus.FAILED.value, error[:2000], video_id),
            )

    def fail_interrupted(self) -> int:
        """Videos left in ``indexing`` by a previous process can never finish."""
        with self._db.connect() as connection:
            cursor = connection.execute(
                "UPDATE videos SET status = ?, error = ? WHERE status = ?",
                (
                    VideoStatus.FAILED.value,
                    "Indexing was interrupted (the app was closed). Index the video again.",
                    VideoStatus.INDEXING.value,
                ),
            )
            return cursor.rowcount

    def delete(self, video_id: str) -> None:
        with self._db.connect() as connection:
            connection.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))


class SceneRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def replace(self, video_id: str, scenes: Sequence[Scene]) -> None:
        with self._db.connect() as connection:
            connection.execute("DELETE FROM scenes WHERE video_id = ?", (video_id,))
            connection.executemany(
                "INSERT INTO scenes (video_id, scene_index, start_ms, end_ms) VALUES (?, ?, ?, ?)",
                [(s.video_id, s.scene_index, s.start_ms, s.end_ms) for s in scenes],
            )

    def for_video(self, video_id: str) -> list[Scene]:
        with self._db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM scenes WHERE video_id = ? ORDER BY scene_index", (video_id,)
            ).fetchall()
        return [Scene(r["video_id"], r["scene_index"], r["start_ms"], r["end_ms"]) for r in rows]


class SegmentRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def replace(self, video_id: str, segments: Sequence[Segment]) -> None:
        with self._db.connect() as connection:
            connection.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
            connection.executemany(
                """
                INSERT INTO segments (
                    segment_id, video_id, seg_index, start_ms, end_ms, preview_frame_path,
                    embedding_model, embedding_dimension, audio_energy, audio_db, motion,
                    motion_raw, scene_cuts, reaction, tags, transcript, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        s.segment_id,
                        s.video_id,
                        s.seg_index,
                        s.start_ms,
                        s.end_ms,
                        s.preview_frame_path,
                        s.embedding_model,
                        s.embedding_dimension,
                        s.audio_energy,
                        s.audio_db,
                        s.motion,
                        s.motion_raw,
                        s.scene_cuts,
                        s.reaction,
                        json.dumps(s.tags, ensure_ascii=False),
                        s.transcript,
                        s.status,
                    )
                    for s in segments
                ],
            )

    def for_video(self, video_id: str) -> list[Segment]:
        with self._db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM segments WHERE video_id = ? ORDER BY seg_index", (video_id,)
            ).fetchall()
        return [_segment(row) for row in rows]

    def get_many(self, segment_ids: Iterable[str]) -> dict[str, Segment]:
        ids = list(dict.fromkeys(segment_ids))
        found: dict[str, Segment] = {}
        with self._db.connect() as connection:
            for offset in range(0, len(ids), 500):
                chunk = ids[offset : offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT * FROM segments WHERE segment_id IN ({placeholders})", chunk
                ).fetchall()
                for row in rows:
                    found[row["segment_id"]] = _segment(row)
        return found

    def overlapping(self, video_id: str, start_ms: int, end_ms: int) -> list[Segment]:
        with self._db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM segments WHERE video_id = ? AND start_ms < ? AND end_ms > ? "
                "ORDER BY seg_index",
                (video_id, end_ms, start_ms),
            ).fetchall()
        return [_segment(row) for row in rows]

    def delete_for_video(self, video_id: str) -> None:
        with self._db.connect() as connection:
            connection.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))


class ClipRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, clip: Clip) -> None:
        with self._db.connect() as connection:
            connection.execute(
                """
                INSERT INTO clips (
                    clip_id, video_id, start_ms, end_ms, query, title, score, score_breakdown,
                    reason, status, draft_path, output_path, thumbnail_path, review_note,
                    created_at, updated_at, exported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clip.clip_id,
                    clip.video_id,
                    clip.start_ms,
                    clip.end_ms,
                    clip.query,
                    clip.title,
                    clip.score,
                    json.dumps(clip.score_breakdown, ensure_ascii=False),
                    clip.reason,
                    clip.status.value,
                    clip.draft_path,
                    clip.output_path,
                    clip.thumbnail_path,
                    clip.review_note,
                    clip.created_at,
                    clip.updated_at,
                    clip.exported_at,
                ),
            )
            connection.execute(
                "INSERT INTO clip_events (clip_id, from_status, to_status, note, created_at) "
                "VALUES (?, NULL, ?, ?, ?)",
                (clip.clip_id, clip.status.value, "clip created", clip.created_at),
            )

    def get(self, clip_id: str) -> Clip | None:
        with self._db.connect() as connection:
            row = connection.execute("SELECT * FROM clips WHERE clip_id = ?", (clip_id,)).fetchone()
        return _clip(row) if row else None

    def require(self, clip_id: str) -> Clip:
        clip = self.get(clip_id)
        if clip is None:
            raise KeyError(f"clip not found: {clip_id}")
        return clip

    def find(
        self,
        *,
        video_id: str | None = None,
        statuses: Sequence[ClipStatus] | None = None,
    ) -> list[Clip]:
        clauses: list[str] = []
        params: list[str] = []
        if video_id is not None:
            clauses.append("video_id = ?")
            params.append(video_id)
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            params.extend(status.value for status in statuses)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._db.connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM clips{where} ORDER BY created_at DESC, clip_id", params
            ).fetchall()
        return [_clip(row) for row in rows]

    def set_status(
        self, clip_id: str, from_status: ClipStatus, to_status: ClipStatus, note: str = ""
    ) -> None:
        now = utc_now()
        with self._db.connect() as connection:
            cursor = connection.execute(
                "UPDATE clips SET status = ?, updated_at = ?, review_note = ? "
                "WHERE clip_id = ? AND status = ?",
                (to_status.value, now, note, clip_id, from_status.value),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"clip {clip_id} changed state during the operation")
            connection.execute(
                "INSERT INTO clip_events (clip_id, from_status, to_status, note, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (clip_id, from_status.value, to_status.value, note, now),
            )

    def update_interval(self, clip_id: str, start_ms: int, end_ms: int) -> None:
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE clips SET start_ms = ?, end_ms = ?, updated_at = ?, draft_path = NULL "
                "WHERE clip_id = ?",
                (start_ms, end_ms, utc_now(), clip_id),
            )

    def set_title(self, clip_id: str, title: str) -> None:
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE clips SET title = ?, updated_at = ? WHERE clip_id = ?",
                (title, utc_now(), clip_id),
            )

    def set_draft(self, clip_id: str, draft_path: str | None) -> None:
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE clips SET draft_path = ?, updated_at = ? WHERE clip_id = ?",
                (draft_path, utc_now(), clip_id),
            )

    def set_export(self, clip_id: str, output_path: str, thumbnail_path: str | None) -> None:
        now = utc_now()
        with self._db.connect() as connection:
            connection.execute(
                "UPDATE clips SET output_path = ?, thumbnail_path = ?, exported_at = ?, "
                "updated_at = ? WHERE clip_id = ?",
                (output_path, thumbnail_path, now, now, clip_id),
            )

    def events(self, clip_id: str) -> list[ClipEvent]:
        with self._db.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM clip_events WHERE clip_id = ? ORDER BY id", (clip_id,)
            ).fetchall()
        return [
            ClipEvent(
                clip_id=row["clip_id"],
                from_status=ClipStatus(row["from_status"]) if row["from_status"] else None,
                to_status=ClipStatus(row["to_status"]),
                note=row["note"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete(self, clip_id: str) -> None:
        with self._db.connect() as connection:
            connection.execute("DELETE FROM clips WHERE clip_id = ?", (clip_id,))


class SearchLogRepository:
    """Local-only metrics for the 'tempo de resposta da busca' KPI."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, video_id: str | None, query: str, results_count: int, elapsed_ms: float) -> None:
        with self._db.connect() as connection:
            connection.execute(
                "INSERT INTO searches (video_id, query, results_count, elapsed_ms, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (video_id, query, results_count, elapsed_ms, utc_now()),
            )

    def stats(self) -> tuple[int, float]:
        """Return ``(number_of_searches, average_latency_ms)``."""
        with self._db.connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n, COALESCE(AVG(elapsed_ms), 0) AS avg_ms FROM searches"
            ).fetchone()
        return int(row["n"]), float(row["avg_ms"])
