"""Optional local speech-to-text for burned-in subtitles (faster-whisper).

Not part of indexing and not an embedding task: it only runs when the user asks
for subtitles on an export, and only on that clip's audio. Install with
``pip install "gta6-clipping-machine[subtitles]"``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from clipping_machine.clipping.subtitle_burner import Cue
from clipping_machine.config import SubtitleConfig


class TranscriberUnavailableError(RuntimeError):
    pass


def is_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


class Transcriber:
    def __init__(self, config: SubtitleConfig, *, local_files_only: bool = False) -> None:
        self._config = config
        self._local_files_only = local_files_only
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            if not is_available():
                raise TranscriberUnavailableError(
                    "Subtitles need the optional dependency: "
                    'pip install "gta6-clipping-machine[subtitles]"'
                )
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._config.model,
                device="cpu",
                compute_type=self._config.compute_type,
                local_files_only=self._local_files_only,
            )
        return self._model

    def transcribe(self, wav_path: Path) -> list[Cue]:
        """Transcribe a WAV file into cues timed from the start of that file."""
        model = self._load()
        segments, _info = model.transcribe(
            str(wav_path), language=self._config.language, vad_filter=True
        )
        cues: list[Cue] = []
        for segment in segments:
            text = str(segment.text).strip()
            if text:
                cues.append(Cue(round(segment.start * 1000), round(segment.end * 1000), text))
        return cues
