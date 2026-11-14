"""Local-only guarantees.

The app never uploads video, audio, frames or embeddings. The only network access
it can ever trigger is the one-time download of model weights from Hugging Face,
and that can be turned off with ``GCM_OFFLINE=1`` once the weights are cached.

This module must be imported before any third-party library that phones home.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

_TRUE = {"1", "true", "yes", "on"}

# Opt-outs for every dependency known to collect usage data.
_TELEMETRY_OPT_OUTS: dict[str, str] = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    "ANONYMIZED_TELEMETRY": "false",
    "TOKENIZERS_PARALLELISM": "false",
}

_OFFLINE_FLAGS: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def is_offline(env: MutableMapping[str, str] | None = None) -> bool:
    """Whether the user asked for strict offline mode (no model downloads)."""
    source = os.environ if env is None else env
    return source.get("GCM_OFFLINE", "").strip().lower() in _TRUE


def enforce_local_only(env: MutableMapping[str, str] | None = None) -> None:
    """Disable telemetry everywhere and, in offline mode, block model downloads."""
    target = os.environ if env is None else env
    for key, value in _TELEMETRY_OPT_OUTS.items():
        target[key] = value
    if is_offline(target):
        for key, value in _OFFLINE_FLAGS.items():
            target[key] = value
