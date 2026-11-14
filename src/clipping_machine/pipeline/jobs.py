"""Runs indexing in a background thread so the UI stays responsive.

One worker: videos are indexed one at a time (the embedding model is the
bottleneck and does not benefit from concurrent jobs). Progress lives in SQLite,
so any browser tab can display it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger(__name__)


class JobManager:
    def __init__(self, run: Callable[[str], object]) -> None:
        self._run = run
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexer")
        self._futures: dict[str, Future[object]] = {}
        self._lock = threading.Lock()

    def submit(self, video_id: str) -> bool:
        """Queue a video. Returns False if it is already queued or running."""
        with self._lock:
            current = self._futures.get(video_id)
            if current is not None and not current.done():
                return False
            self._futures[video_id] = self._executor.submit(self._guarded, video_id)
            return True

    def _guarded(self, video_id: str) -> object:
        try:
            return self._run(video_id)
        except Exception:
            # The indexer already recorded the failure on the video row.
            logger.exception("indexing failed for %s", video_id)
            raise

    def is_active(self, video_id: str) -> bool:
        with self._lock:
            future = self._futures.get(video_id)
            return future is not None and not future.done()

    def active_ids(self) -> list[str]:
        with self._lock:
            return [vid for vid, future in self._futures.items() if not future.done()]

    def wait(self, video_id: str, timeout: float | None = None) -> None:
        """Block until a job finishes (used by tests and the CLI)."""
        with self._lock:
            future = self._futures.get(video_id)
        if future is not None:
            future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
