"""Helpers shared by the screens: translation, formatting, reasons."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import streamlit as st

from clipping_machine.config import Settings
from clipping_machine.ingest.video_validator import ValidationIssue
from clipping_machine.ui import i18n

Translator = Callable[..., str]


def current_language() -> str:
    return str(st.session_state.get("language", i18n.DEFAULT_LANGUAGE))


def translator() -> Translator:
    language = current_language()

    def t(key: str, **params: object) -> str:
        return i18n.translate(language, key, **params)

    return t


def format_time(ms: int) -> str:
    """``M:SS`` or ``H:MM:SS``."""
    total = max(0, int(ms)) // 1000
    hours, minutes, seconds = total // 3600, total % 3600 // 60, total % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def format_range(start_ms: int, end_ms: int) -> str:
    return f"{format_time(start_ms)}–{format_time(end_ms)}"


def format_score(value: float) -> str:
    text = f"{value:.2f}"
    return text.replace(".", ",") if current_language() == "pt" else text


def format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def tag_labels(settings: Settings) -> dict[str, str]:
    """Concept id → label in the current language."""
    portuguese = current_language() == "pt"
    return {
        concept.id: (concept.label_pt or concept.label) if portuguese else concept.label
        for concept in settings.tags.concepts
    }


def describe_reason(
    tags: Sequence[str], highlights: Sequence[str], labels: Mapping[str, str], t: Translator
) -> str:
    """Human-readable "why this moment": zero-shot tags plus the notable signals."""
    parts: list[str] = []
    named = [labels.get(tag, tag) for tag in tags]
    if named:
        parts.append(", ".join(named))
    parts.extend(t(f"highlight.{code}") for code in highlights)
    return " · ".join(parts) if parts else t("reason.fallback")


def issue_messages(issues: Sequence[ValidationIssue]) -> list[str]:
    if current_language() == "pt":
        return [i18n.ISSUE_MESSAGES_PT.get(issue.code, issue.message) for issue in issues]
    return [issue.message for issue in issues]
