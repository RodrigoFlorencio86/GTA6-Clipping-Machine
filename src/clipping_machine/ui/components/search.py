"""Search screen: query, timeline, moment cards, manual trim, clip creation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from clipping_machine import ffmpeg
from clipping_machine.analysis.highlight_scorer import SIGNAL_NAMES
from clipping_machine.embeddings.embeddinggemma_client import EmbeddingModelError
from clipping_machine.search.moment_finder import Moment, SearchResult
from clipping_machine.services import ClippingService, ClipTooLongError
from clipping_machine.storage.models import Video, VideoStatus
from clipping_machine.ui.components.common import (
    Translator,
    describe_reason,
    format_range,
    format_score,
    format_time,
    tag_labels,
    translator,
)

_ALL = "__all__"
# How far (seconds) the trim slider lets the user extend a moment on each side.
_TRIM_MARGIN_S = 30.0


def _search_form(service: ClippingService, videos: list[Video], t: Translator) -> None:
    config = service.settings.search
    names = {video.video_id: video.original_name for video in videos}
    with st.form("search_form", border=True):
        left, right = st.columns([1, 2])
        video_choice = left.selectbox(
            t("search.video"),
            options=[_ALL, *names],
            format_func=lambda vid: t("search.all_videos") if vid == _ALL else names[vid],
            key="search_video",
        )
        query = right.text_input(
            t("search.query"), placeholder=t("search.placeholder"), key="search_query"
        )
        with st.expander(t("search.advanced")):
            a, b, c = st.columns(3)
            min_similarity = a.slider(
                t("search.min_similarity"),
                0.0,
                1.0,
                float(config.min_similarity),
                0.01,
                help=t("search.min_similarity_help"),
                key="search_min_similarity",
            )
            top_k = b.number_input(
                t("search.top_k"), 5, 500, int(config.top_k), 5, key="search_top_k"
            )
            merge_gap = c.number_input(
                t("search.merge_gap"),
                0.0,
                30.0,
                float(config.merge_gap_seconds),
                0.5,
                key="search_merge_gap",
            )
            fill = st.checkbox(
                t("search.fill", n=config.min_results), value=True, key="search_fill"
            )
        submitted = st.form_submit_button(t("search.button"), type="primary")

    if not submitted:
        return
    if not query.strip():
        st.warning(t("search.query"))
        return
    message = t("search.searching") if _model_loaded(service) else t("search.loading_model")
    try:
        with st.spinner(message):
            result = service.search(
                query,
                video_id=None if video_choice == _ALL else video_choice,
                top_k=int(top_k),
                min_similarity=float(min_similarity),
                merge_gap_seconds=float(merge_gap),
                fill_to_min_results=fill,
            )
    except (EmbeddingModelError, ValueError) as exc:
        st.error(t("common.error", error=exc))
        return
    st.session_state["search_result"] = result
    st.session_state.pop("search_selected", None)


def _model_loaded(service: ClippingService) -> bool:
    return bool(getattr(service.backend, "is_loaded", True))


def _timeline(
    service: ClippingService, video: Video, moments: list[tuple[int, Moment]], t: Translator
) -> None:
    """Moments as bars over the video's duration, with the audio/motion curves below."""
    rows = [
        {
            "rank": rank,
            "start": moment.start_ms / 1000,
            "end": moment.end_ms / 1000,
            "score": round(moment.score, 3),
            "range": format_range(moment.start_ms, moment.end_ms),
            "key": moment.key,
        }
        for rank, moment in moments
    ]
    duration_s = video.duration_ms / 1000
    scale = alt.Scale(domain=[0, duration_s], nice=False)
    pick = alt.selection_point(name="moment", fields=["key"], on="click", empty=False)
    bars = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(height=26, cornerRadius=3, stroke="white", strokeWidth=1)
        .encode(
            x=alt.X("start:Q", scale=scale, title=None, axis=alt.Axis(labels=False, ticks=False)),
            x2="end:Q",
            color=alt.Color(
                "score:Q",
                scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                legend=alt.Legend(title=t("search.score"), orient="right"),
            ),
            opacity=alt.condition(pick, alt.value(1.0), alt.value(0.85)),
            tooltip=[
                alt.Tooltip("rank:Q", title="#"),
                alt.Tooltip("range:N", title=t("search.timeline")),
                alt.Tooltip("score:Q", title=t("search.score"), format=".2f"),
            ],
        )
        .add_params(pick)
        .properties(height=70)
    )
    event = st.altair_chart(
        bars, on_select="rerun", selection_mode="moment", key=f"timeline_{video.video_id}"
    )
    try:
        picked = event["selection"]["moment"]  # type: ignore[index]
    except (KeyError, TypeError):
        picked = []
    if picked:
        st.session_state["search_selected"] = picked[0].get("key")

    segments = service.segments.for_video(video.video_id)
    if segments:
        frame = pd.DataFrame(
            [
                {
                    "t": (segment.start_ms + segment.end_ms) / 2000,
                    t("signal.audio_intensity"): segment.audio_energy,
                    t("signal.visual_motion"): segment.motion,
                }
                for segment in segments
            ]
        ).melt("t", var_name="signal", value_name="value")
        lines = (
            alt.Chart(frame)
            .mark_line(interpolate="monotone", strokeWidth=1.5)
            .encode(
                x=alt.X(
                    "t:Q",
                    scale=scale,
                    title=None,
                    axis=alt.Axis(
                        labelExpr="floor(datum.value / 60) + ':' + "
                        "(datum.value % 60 < 10 ? '0' : '') + floor(datum.value % 60)"
                    ),
                ),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                color=alt.Color("signal:N", legend=alt.Legend(title=None, orient="right")),
            )
            .properties(height=110)
        )
        st.altair_chart(lines)
        st.caption(t("search.signals"))


def _breakdown(service: ClippingService, moment: Moment, t: Translator) -> None:
    weights = service.settings.scoring.model_dump()
    signals = moment.breakdown.as_dict()
    table = pd.DataFrame(
        [
            {
                "signal": t(f"signal.{name}"),
                t("signal.weight"): weights[name],
                "value": signals[name],
                "points": round(weights[name] * signals[name], 3),
            }
            for name in SIGNAL_NAMES
        ]
    )
    st.dataframe(table, hide_index=True, width="stretch")


def _moment_card(
    service: ClippingService,
    video: Video,
    rank: int,
    moment: Moment,
    result: SearchResult,
    labels: Mapping[str, str],
    t: Translator,
) -> None:
    max_s = service.settings.search.max_clip_duration_seconds
    selected = st.session_state.get("search_selected") == moment.key
    reason = describe_reason(moment.tags, moment.highlights, labels, t)
    with st.container(border=True):
        picture, body, actions = st.columns([1.1, 2.4, 1.5])
        with picture:
            if moment.preview_frame_path and Path(moment.preview_frame_path).is_file():
                st.image(moment.preview_frame_path, width="stretch")
        with body:
            badges = []
            if selected:
                badges.append(f":blue-badge[{t('search.selected')}]")
            if moment.below_threshold:
                badges.append(f":orange-badge[{t('search.below_threshold')}]")
            st.markdown(
                f"**{t('search.moment', rank=rank)}** · "
                f"{format_range(moment.start_ms, moment.end_ms)} "
                f"({moment.duration_ms / 1000:.0f} s) {' '.join(badges)}"
            )
            st.caption(video.original_name)
            one, two = st.columns(2)
            one.metric(t("search.score"), format_score(moment.score))
            two.metric(t("search.similarity"), format_score(moment.similarity))
            st.markdown(f"**{t('search.reason')}:** {reason}")
            with st.expander(t("search.breakdown")):
                _breakdown(service, moment, t)
        with actions:
            # Keep everything on the slider's 0.5 s grid.
            video_s = video.duration_ms / 1000
            low = float(math.floor(max(0.0, moment.start_ms / 1000 - _TRIM_MARGIN_S)))
            high = math.ceil(min(video_s, moment.end_ms / 1000 + _TRIM_MARGIN_S) * 2) / 2
            start_s, end_s = st.slider(
                t("search.adjust"),
                min_value=low,
                max_value=high,
                value=(
                    max(low, round(moment.start_ms / 500) / 2),
                    min(high, round(moment.end_ms / 500) / 2),
                ),
                step=0.5,
                key=f"trim_{moment.key}",
            )
            start_ms = int(start_s * 1000)
            end_ms = min(video.duration_ms, int(end_s * 1000))
            st.caption(f"{format_time(start_ms)}–{format_time(end_ms)} · {end_s - start_s:.1f} s")
            too_long = end_s - start_s > max_s
            invalid = end_ms <= start_ms
            if too_long:
                st.error(t("search.too_long", max=max_s))

            if st.button(
                t("search.preview"),
                key=f"preview_{moment.key}",
                width="stretch",
                disabled=too_long or invalid,
            ):
                try:
                    with st.spinner(t("search.preview_rendering")):
                        clip_path = service.render_preview(video.video_id, start_ms, end_ms)
                    st.session_state[f"preview_path_{moment.key}"] = str(clip_path)
                except (ffmpeg.FFmpegError, ValueError) as exc:
                    st.error(t("common.error", error=exc))
            preview_path = st.session_state.get(f"preview_path_{moment.key}")
            if preview_path and Path(preview_path).is_file():
                st.video(preview_path)

            if st.button(
                t("search.generate"),
                key=f"generate_{moment.key}",
                type="primary",
                width="stretch",
                disabled=too_long or invalid,
            ):
                try:
                    with st.spinner(t("search.generating")):
                        clip = service.create_clip(
                            moment,
                            query=result.query,
                            reason=reason,
                            start_ms=start_ms,
                            end_ms=end_ms,
                        )
                        service.render_draft(clip.clip_id)
                    st.toast(t("search.generated"))
                except (ClipTooLongError, ValueError, ffmpeg.FFmpegError) as exc:
                    st.error(t("common.error", error=exc))


def render(service: ClippingService) -> None:
    t = translator()
    videos = service.videos.all(VideoStatus.INDEXED)
    if not videos:
        st.info(t("search.no_videos"))
        return
    _search_form(service, videos, t)

    result: SearchResult | None = st.session_state.get("search_result")
    if result is None:
        return
    if not result.moments:
        st.warning(t("search.none"))
        return

    summary = t(
        "search.summary",
        moments=len(result.moments),
        hits=result.hits_above_threshold,
        ms=result.elapsed_ms,
    )
    if result.filled:
        summary += " " + t("search.summary_filled", filled=result.filled)
    st.caption(summary)

    max_s = service.settings.search.max_clip_duration_seconds
    left, right = st.columns(2)
    min_score = left.slider(t("search.filter_score"), 0.0, 1.0, 0.0, 0.05, key="filter_score")
    duration = right.slider(
        t("search.filter_duration"),
        0.0,
        float(max_s),
        (0.0, float(max_s)),
        1.0,
        key="filter_duration",
    )

    by_id = {video.video_id: video for video in videos}
    ranked = [
        (rank, moment)
        for rank, moment in enumerate(result.moments, start=1)
        if moment.video_id in by_id
        and moment.score >= min_score
        and duration[0] <= moment.duration_ms / 1000 <= duration[1]
    ]
    if not ranked:
        st.warning(t("search.none_after_filter"))
        return

    labels = tag_labels(service.settings)
    st.subheader(t("search.timeline"))
    st.caption(t("search.timeline_help"))
    for video_id in dict.fromkeys(moment.video_id for _, moment in ranked):
        video = by_id[video_id]
        if len(by_id) > 1:
            st.markdown(f"**{video.original_name}**")
        _timeline(service, video, [(r, m) for r, m in ranked if m.video_id == video_id], t)

    selected_key = st.session_state.get("search_selected")
    ordered = sorted(ranked, key=lambda item: item[1].key != selected_key)
    for rank, moment in ordered:
        _moment_card(service, by_id[moment.video_id], rank, moment, result, labels, t)
