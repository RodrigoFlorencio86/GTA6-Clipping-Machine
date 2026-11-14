"""Assembles per-window model inputs and embeds them in batches."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from clipping_machine.embeddings.base import EmbeddingBackend, Matrix, WindowInput
from clipping_machine.ingest.audio_chunker import AudioTrack
from clipping_machine.ingest.frame_extractor import FrameExtraction
from clipping_machine.ingest.windows import Window, keyframes_for_window


def build_window_input(
    window: Window, frames: FrameExtraction, audio: AudioTrack | None
) -> WindowInput:
    """Keyframes on the sampling grid inside the window, plus its audio chunk."""
    indices = keyframes_for_window(window, frames.keyframe_interval_ms, len(frames.keyframes))
    samples = audio.read(window.start_ms, window.end_ms) if audio is not None else None
    return WindowInput(
        frames=tuple(frames.keyframes[index] for index in indices),
        audio=samples,
        sample_rate=audio.sample_rate if audio is not None else 16000,
    )


def embed_in_batches(
    backend: EmbeddingBackend,
    inputs: Sequence[WindowInput],
    batch_size: int,
    on_progress: Callable[[float], None] | None = None,
) -> Matrix:
    """Embed windows chunk by chunk so progress can be reported and memory stays flat."""
    if not inputs:
        return np.zeros((0, backend.dimension), dtype=np.float32)
    chunks: list[Matrix] = []
    for offset in range(0, len(inputs), batch_size):
        chunks.append(backend.embed_windows(inputs[offset : offset + batch_size]))
        if on_progress is not None:
            on_progress(min(1.0, (offset + batch_size) / len(inputs)))
    return np.concatenate(chunks, axis=0)
