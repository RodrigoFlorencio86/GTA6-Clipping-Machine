"""SQLite connection handling and schema."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id            TEXT PRIMARY KEY,
    original_name       TEXT NOT NULL,
    path                TEXT NOT NULL,
    managed_copy        INTEGER NOT NULL,
    size_bytes          INTEGER NOT NULL,
    fingerprint         TEXT NOT NULL,
    duration_ms         INTEGER NOT NULL,
    width               INTEGER NOT NULL,
    height              INTEGER NOT NULL,
    fps                 REAL NOT NULL,
    video_codec         TEXT NOT NULL,
    audio_codec         TEXT,
    rights_confirmed_at TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    status              TEXT NOT NULL,
    stage               TEXT NOT NULL DEFAULT '',
    progress            REAL NOT NULL DEFAULT 0,
    error               TEXT,
    segments_count      INTEGER NOT NULL DEFAULT 0,
    indexed_at          TEXT,
    index_seconds       REAL,
    embedding_model     TEXT
);
CREATE INDEX IF NOT EXISTS idx_videos_fingerprint ON videos(fingerprint);

CREATE TABLE IF NOT EXISTS scenes (
    video_id    TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    scene_index INTEGER NOT NULL,
    start_ms    INTEGER NOT NULL,
    end_ms      INTEGER NOT NULL,
    PRIMARY KEY (video_id, scene_index)
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id          TEXT PRIMARY KEY,
    video_id            TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    seg_index           INTEGER NOT NULL,
    start_ms            INTEGER NOT NULL,
    end_ms              INTEGER NOT NULL,
    preview_frame_path  TEXT NOT NULL,
    embedding_model     TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    audio_energy        REAL NOT NULL DEFAULT 0,
    audio_db            REAL NOT NULL DEFAULT -80,
    motion              REAL NOT NULL DEFAULT 0,
    motion_raw          REAL NOT NULL DEFAULT 0,
    scene_cuts          INTEGER NOT NULL DEFAULT 0,
    reaction            REAL NOT NULL DEFAULT 0,
    tags                TEXT NOT NULL DEFAULT '[]',
    transcript          TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'indexed'
);
CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id, start_ms);

CREATE TABLE IF NOT EXISTS clips (
    clip_id         TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    start_ms        INTEGER NOT NULL,
    end_ms          INTEGER NOT NULL,
    query           TEXT NOT NULL DEFAULT '',
    title           TEXT NOT NULL DEFAULT '',
    score           REAL NOT NULL DEFAULT 0,
    score_breakdown TEXT NOT NULL DEFAULT '{}',
    reason          TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL,
    draft_path      TEXT,
    output_path     TEXT,
    thumbnail_path  TEXT,
    review_note     TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    exported_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_clips_video ON clips(video_id);
CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);

CREATE TABLE IF NOT EXISTS clip_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_id     TEXT NOT NULL REFERENCES clips(clip_id) ON DELETE CASCADE,
    from_status TEXT,
    to_status   TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

-- Local-only usage metrics (never transmitted): search latency and volume.
CREATE TABLE IF NOT EXISTS searches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT,
    query         TEXT NOT NULL,
    results_count INTEGER NOT NULL,
    elapsed_ms    REAL NOT NULL,
    created_at    TEXT NOT NULL
);
"""


def utc_now() -> str:
    """ISO-8601 UTC timestamp with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class Database:
    """Opens short-lived connections so it is safe to share across threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
