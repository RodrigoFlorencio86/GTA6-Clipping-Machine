"""Loudness measurements for audio chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from clipping_machine.analysis.normalization import robust_normalize

SILENCE_DB = -80.0
# Below this loudness the whole video is treated as silent (no audio signal to rank).
AUDIBLE_FLOOR_DB = -60.0


@dataclass(frozen=True)
class AudioStats:
    """Loudness of one chunk, in dBFS."""

    mean_db: float
    peak_db: float

    @property
    def intensity_db(self) -> float:
        """Blend of sustained loudness and short peaks (explosions, shots, shouts)."""
        return 0.5 * self.mean_db + 0.5 * self.peak_db


SILENT = AudioStats(SILENCE_DB, SILENCE_DB)


def to_db(rms: float) -> float:
    if rms <= 0:
        return SILENCE_DB
    return max(SILENCE_DB, 20.0 * float(np.log10(rms)))


def short_time_rms(
    samples: NDArray[np.float32], sample_rate: int, frame_ms: int = 50
) -> NDArray[np.float32]:
    """RMS of consecutive ``frame_ms`` frames."""
    frame = max(1, int(sample_rate * frame_ms / 1000))
    usable = (samples.size // frame) * frame
    if usable == 0:
        if samples.size == 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray([np.sqrt(np.mean(np.square(samples, dtype=np.float64)))], np.float32)
    frames = samples[:usable].astype(np.float64).reshape(-1, frame)
    return np.sqrt(np.mean(np.square(frames), axis=1)).astype(np.float32)


def measure(samples: NDArray[np.float32], sample_rate: int) -> AudioStats:
    """Mean loudness and 95th-percentile short-time loudness of a chunk."""
    if samples.size == 0:
        return SILENT
    mean_rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    frames = short_time_rms(samples, sample_rate)
    peak_rms = float(np.percentile(frames, 95)) if frames.size else mean_rms
    return AudioStats(to_db(mean_rms), to_db(peak_rms))


def normalize_intensity(stats: Sequence[AudioStats]) -> NDArray[np.float32]:
    """Per-video audio intensity in ``[0, 1]`` for each chunk.

    Relative to the video's own loudness distribution, so a quiet recording and a
    loud one are ranked comparably. A silent video yields zeros.
    """
    if not stats:
        return np.zeros(0, dtype=np.float32)
    raw = np.asarray([item.intensity_db for item in stats], dtype=np.float64)
    if float(np.max(raw)) < AUDIBLE_FLOOR_DB:
        return np.zeros(raw.shape, dtype=np.float32)
    return robust_normalize(raw, min_range=1.0)
