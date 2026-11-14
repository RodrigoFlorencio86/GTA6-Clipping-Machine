"""Status screen: environment checks, local metrics, privacy notice."""

from __future__ import annotations

import streamlit as st

from clipping_machine import ffmpeg, privacy
from clipping_machine.embeddings.embeddinggemma_client import EmbeddingModelError
from clipping_machine.services import ClippingService
from clipping_machine.storage.models import ClipStatus, VideoStatus
from clipping_machine.ui.components.common import translator


def render(service: ClippingService) -> None:
    t = translator()
    st.subheader(t("about.environment"))
    backend = service.backend
    loaded = bool(getattr(backend, "is_loaded", True))
    device = getattr(backend, "device", None) or "?"

    left, right = st.columns(2)
    with left:
        st.markdown(
            f"**{t('about.ffmpeg')}:** "
            f"{t('common.ok') if ffmpeg.is_available() else t('common.missing')}"
        )
        model_state = (
            t("about.model_loaded", device=device) if loaded else t("about.model_not_loaded")
        )
        st.markdown(
            f"**{t('about.model')}:** `{backend.model_name}` ({backend.dimension}d), {model_state}"
        )
        if not loaded and st.button(t("about.load_model")):
            try:
                with st.spinner(t("search.loading_model")):
                    backend.load()  # type: ignore[attr-defined]
            except EmbeddingModelError as exc:
                st.error(t("common.error", error=exc))
            else:
                st.rerun()
    with right:
        st.markdown(f"**{t('about.data_dir')}:** `{service.settings.data_dir}`")
        st.markdown(f"**{t('about.vectors')}:** {service.store.count()}")
        offline = t("common.yes") if privacy.is_offline() else t("common.no")
        st.markdown(f"**{t('about.offline')}:** {offline}")

    st.subheader(t("about.metrics"))
    searches, latency = service.search_log.stats()
    clips = service.clips.find()
    approved = sum(
        1
        for clip in clips
        if clip.status in (ClipStatus.APPROVED, ClipStatus.EXPORTED, ClipStatus.PUBLISHED)
    )
    rejected = sum(1 for clip in clips if clip.status is ClipStatus.REJECTED)
    exported = sum(
        1 for clip in clips if clip.status in (ClipStatus.EXPORTED, ClipStatus.PUBLISHED)
    )
    indexed = [v for v in service.videos.all(VideoStatus.INDEXED) if v.index_seconds]
    video_seconds = sum(v.duration_ms for v in indexed) / 1000
    index_seconds = sum(v.index_seconds or 0.0 for v in indexed)

    columns = st.columns(5)
    columns[0].metric(t("about.metric_searches"), searches)
    columns[1].metric(t("about.metric_latency"), f"{latency:.0f} ms" if searches else "—")
    decided = approved + rejected
    columns[2].metric(
        t("about.metric_approval"), f"{approved / decided * 100:.0f}%" if decided else "—"
    )
    columns[3].metric(t("about.metric_exported"), exported)
    columns[4].metric(
        t("about.metric_speed"),
        t("about.metric_speed_value", ratio=index_seconds / video_seconds)
        if video_seconds
        else "—",
    )

    st.subheader(t("about.privacy"))
    st.markdown(t("about.privacy_body"))
