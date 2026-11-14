"""Approval queue screen: review, approve/reject, export, bookkeeping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from clipping_machine import ffmpeg
from clipping_machine.analysis import transcriber
from clipping_machine.analysis.highlight_scorer import SIGNAL_NAMES
from clipping_machine.publishing.approval_queue import InvalidTransitionError
from clipping_machine.services import ClippingService, ClipTooLongError
from clipping_machine.storage.models import Clip, ClipStatus
from clipping_machine.ui.components.common import (
    Translator,
    format_range,
    format_score,
    translator,
)

_FRAMINGS = ("smart", "center", "blur_pad")
_ORDER = (
    ClipStatus.CANDIDATE,
    ClipStatus.APPROVED,
    ClipStatus.EXPORTED,
    ClipStatus.PUBLISHED,
    ClipStatus.REJECTED,
    ClipStatus.ARCHIVED,
)


_PENDING_FILTER = "queue_status_next"


def _follow(status: ClipStatus) -> None:
    """Show ``status`` in the filter on the next run.

    The filter widget already exists in this run, so its key cannot be changed now.
    """
    st.session_state[_PENDING_FILTER] = status


def _act(t: Translator, action: object, follow: ClipStatus | None = None) -> None:
    """Run a queue action and refresh; show a readable error if it is not allowed.

    ``follow`` switches the status filter so the clip stays in view after it moves.
    """
    try:
        action()  # type: ignore[operator]
    except (InvalidTransitionError, ClipTooLongError, ValueError, RuntimeError, OSError) as exc:
        st.error(t("common.error", error=exc))
    else:
        if follow is not None:
            _follow(follow)
        st.rerun()


def _player(service: ClippingService, clip: Clip, t: Translator) -> None:
    final = clip.output_path if clip.status in (ClipStatus.EXPORTED, ClipStatus.PUBLISHED) else None
    source = final or clip.draft_path
    if source and Path(source).is_file():
        st.video(source)
    else:
        st.caption(t("queue.no_draft"))
    if clip.status in (ClipStatus.CANDIDATE, ClipStatus.APPROVED) and st.button(
        t("queue.render_draft"), key=f"draft_{clip.clip_id}", width="stretch"
    ):
        try:
            with st.spinner(t("queue.rendering")):
                service.render_draft(clip.clip_id)
        except (ffmpeg.FFmpegError, ValueError) as exc:
            st.error(t("common.error", error=exc))
        else:
            st.rerun()


def _details(service: ClippingService, clip: Clip, t: Translator) -> None:
    video = service.videos.get(clip.video_id)
    name = video.original_name if video else clip.video_id
    st.markdown(
        f"**{clip.title or name}** · {format_range(clip.start_ms, clip.end_ms)} "
        f"({clip.duration_ms / 1000:.0f} s)"
    )
    st.caption(f"{name} · {t(f'clip.{clip.status.value}')}")
    left, right = st.columns(2)
    left.metric(t("search.score"), format_score(clip.score))
    if clip.query:
        right.markdown(f"**{t('queue.query')}:** {clip.query}")
    if clip.reason:
        st.markdown(f"**{t('search.reason')}:** {clip.reason}")
    if clip.output_path and clip.status in (ClipStatus.EXPORTED, ClipStatus.PUBLISHED):
        st.markdown(f"**{t('queue.file')}:** `{clip.output_path}`")

    with st.expander(t("search.breakdown")):
        weights = service.settings.scoring.model_dump()
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "signal": t(f"signal.{name_}"),
                        t("signal.weight"): weights[name_],
                        "value": clip.score_breakdown.get(name_, 0.0),
                    }
                    for name_ in SIGNAL_NAMES
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander(t("queue.history")):
        for event in service.clips.events(clip.clip_id):
            origin = t(f"clip.{event.from_status.value}") if event.from_status else "—"
            note = f" · {event.note}" if event.note else ""
            st.caption(
                f"{event.created_at} · {origin} → {t(f'clip.{event.to_status.value}')}{note}"
            )


def _adjust(service: ClippingService, clip: Clip, t: Translator) -> None:
    video = service.videos.get(clip.video_id)
    if video is None:
        return
    with st.popover(t("queue.adjust"), width="stretch"):
        start = st.number_input(
            t("queue.start"),
            0.0,
            video.duration_ms / 1000,
            clip.start_ms / 1000,
            0.5,
            key=f"start_{clip.clip_id}",
        )
        end = st.number_input(
            t("queue.end"),
            0.0,
            video.duration_ms / 1000,
            clip.end_ms / 1000,
            0.5,
            key=f"end_{clip.clip_id}",
        )
        if st.button(t("queue.apply"), key=f"apply_{clip.clip_id}"):
            _act(
                t,
                lambda: service.adjust_clip(clip.clip_id, int(start * 1000), int(end * 1000)),
            )


def _export_controls(service: ClippingService, clip: Clip, t: Translator) -> None:
    framing = st.selectbox(
        t("queue.framing"),
        _FRAMINGS,
        index=_FRAMINGS.index(service.settings.export.framing),
        format_func=lambda value: t(f"framing.{value}"),
        key=f"framing_{clip.clip_id}",
    )
    subtitles_ready = transcriber.is_available()
    subtitles = st.checkbox(
        t("queue.subtitles"),
        value=False,
        disabled=not subtitles_ready,
        help=None if subtitles_ready else t("queue.subtitles_unavailable"),
        key=f"subtitles_{clip.clip_id}",
    )
    if st.button(t("queue.export"), key=f"export_{clip.clip_id}", type="primary", width="stretch"):
        bar = st.progress(0.0, text=t("queue.exporting"))
        try:
            result = service.export_clip(
                clip.clip_id,
                framing=framing,  # type: ignore[arg-type]
                burn_subtitles=subtitles,
                on_progress=lambda fraction: bar.progress(
                    min(1.0, fraction), text=t("queue.exporting")
                ),
            )
        except (ffmpeg.FFmpegError, InvalidTransitionError, RuntimeError, ValueError) as exc:
            bar.empty()
            st.error(t("common.error", error=exc))
        else:
            st.session_state["queue_notice"] = t("queue.exported", path=result.output_path)
            _follow(ClipStatus.EXPORTED)
            st.rerun()


def _actions(service: ClippingService, clip: Clip, t: Translator) -> None:
    queue = service.queue
    key = clip.clip_id
    status = clip.status
    if status is ClipStatus.CANDIDATE:
        note = st.text_input(t("queue.note"), key=f"note_{key}")
        left, right = st.columns(2)
        if left.button(t("queue.approve"), key=f"approve_{key}", type="primary", width="stretch"):
            _act(t, lambda: queue.approve(key, note), ClipStatus.APPROVED)
        if right.button(t("queue.reject"), key=f"reject_{key}", width="stretch"):
            _act(t, lambda: queue.reject(key, note), ClipStatus.REJECTED)
        _adjust(service, clip, t)
    elif status is ClipStatus.APPROVED:
        _export_controls(service, clip, t)
        _adjust(service, clip, t)
        left, right = st.columns(2)
        if left.button(t("queue.reopen"), key=f"reopen_{key}", width="stretch"):
            _act(t, lambda: queue.reopen(key), ClipStatus.CANDIDATE)
        if right.button(t("queue.reject"), key=f"reject_{key}", width="stretch"):
            _act(t, lambda: queue.reject(key), ClipStatus.REJECTED)
    elif status is ClipStatus.EXPORTED:
        if clip.output_path and Path(clip.output_path).is_file():
            st.download_button(
                t("queue.download"),
                data=Path(clip.output_path).read_bytes(),
                file_name=Path(clip.output_path).name,
                mime="video/mp4",
                key=f"download_{key}",
                width="stretch",
            )
        if st.button(
            t("queue.mark_published"),
            key=f"published_{key}",
            help=t("queue.mark_published_help"),
            width="stretch",
        ):
            _act(t, lambda: queue.mark_published_manually(key), ClipStatus.PUBLISHED)
        left, right = st.columns(2)
        if left.button(t("queue.export_again"), key=f"again_{key}", width="stretch"):
            _act(
                t,
                lambda: queue.transition(key, ClipStatus.APPROVED, "re-export requested"),
                ClipStatus.APPROVED,
            )
        if right.button(t("queue.archive"), key=f"archive_{key}", width="stretch"):
            _act(t, lambda: queue.archive(key), ClipStatus.ARCHIVED)
    elif status is ClipStatus.REJECTED:
        left, right = st.columns(2)
        if left.button(t("queue.reopen"), key=f"reopen_{key}", width="stretch"):
            _act(t, lambda: queue.reopen(key), ClipStatus.CANDIDATE)
        if right.button(t("queue.archive"), key=f"archive_{key}", width="stretch"):
            _act(t, lambda: queue.archive(key), ClipStatus.ARCHIVED)
    elif status is ClipStatus.PUBLISHED and st.button(
        t("queue.archive"), key=f"archive_{key}", width="stretch"
    ):
        _act(t, lambda: queue.archive(key), ClipStatus.ARCHIVED)

    if status in (ClipStatus.CANDIDATE, ClipStatus.REJECTED, ClipStatus.ARCHIVED):
        with st.popover(t("queue.delete"), width="stretch"):
            if st.button(t("library.delete_yes"), key=f"delete_clip_{key}"):
                _act(t, lambda: service.delete_clip(key))


def render(service: ClippingService) -> None:
    t = translator()
    st.info(t("queue.intro"))
    notice = st.session_state.pop("queue_notice", None)
    if notice:
        st.success(notice)

    clips = service.clips.find()
    counts = {status: sum(1 for clip in clips if clip.status is status) for status in _ORDER}
    pending = st.session_state.pop(_PENDING_FILTER, None)
    if pending is not None:
        st.session_state["queue_status"] = pending
    elif "queue_status" not in st.session_state:
        st.session_state["queue_status"] = ClipStatus.CANDIDATE
    chosen = st.segmented_control(
        t("queue.filter"),
        options=list(_ORDER),
        format_func=lambda status: f"{t(f'clip.{status.value}')} ({counts[status]})",
        key="queue_status",
        required=True,
    )
    visible = [clip for clip in clips if clip.status is chosen]
    if not visible:
        st.caption(t("queue.empty"))
        return
    for clip in visible:
        with st.container(border=True):
            player, details, actions = st.columns([1, 2.2, 1.4])
            with player:
                _player(service, clip, t)
            with details:
                _details(service, clip, t)
            with actions:
                _actions(service, clip, t)
