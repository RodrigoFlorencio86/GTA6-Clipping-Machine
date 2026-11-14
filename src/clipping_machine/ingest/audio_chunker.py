"""Audio extraction (mono PCM via FFmpeg) and per-window chunking."""

from __future__ import annotations

import wave
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from clipping_machine import ffmpeg
from clipping_machine.ingest.windows import Window


def extract_audio(
    video_path: Path,
    output_wav: Path,
    sample_rate: int,
    *,
    duration_ms: int | None = None,
    start_ms: int | None = None,
    clip_ms: int | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Path:
    """Decode the first audio stream to 16-bit mono WAV at ``sample_rate``."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = []
    if start_ms is not None:
        args += ["-ss", ffmpeg.seconds(start_ms)]
    args += ["-i", str(video_path)]
    if clip_ms is not None:
        args += ["-t", ffmpeg.seconds(clip_ms)]
    args += [
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    ffmpeg.run_ffmpeg(args, duration_ms=duration_ms, on_progress=on_progress)
    return output_wav


class AudioTrack:
    """Random access to a mono 16-bit WAV file as float32 samples in ``[-1, 1]``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._reader = wave.open(str(path), "rb")  # noqa: SIM115 - closed in close()
        if self._reader.getnchannels() != 1 or self._reader.getsampwidth() != 2:
            self._reader.close()
            raise ValueError("expected a mono 16-bit PCM WAV file")
        self.sample_rate: int = self._reader.getframerate()
        self.n_samples: int = self._reader.getnframes()

    @property
    def duration_ms(self) -> int:
        return int(self.n_samples / self.sample_rate * 1000)

    def read(self, start_ms: int, end_ms: int) -> NDArray[np.float32]:
        """Samples for ``[start_ms, end_ms)``; shorter (or empty) past the end of the track."""
        first = max(0, int(start_ms * self.sample_rate / 1000))
        last = min(self.n_samples, int(end_ms * self.sample_rate / 1000))
        if last <= first:
            return np.zeros(0, dtype=np.float32)
        self._reader.setpos(first)
        raw = self._reader.readframes(last - first)
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> AudioTrack:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def iter_chunks(
    track: AudioTrack, windows: Sequence[Window]
) -> Iterator[tuple[Window, NDArray[np.float32]]]:
    """Yield the audio samples that belong to each window."""
    for window in windows:
        yield window, track.read(window.start_ms, window.end_ms)
