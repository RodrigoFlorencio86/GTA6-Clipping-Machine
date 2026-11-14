"""Videos screen: local import, list, indexing progress, deletion."""

from __future__ import annotations

import streamlit as st

from clipping_machine import ffmpeg
from clipping_machine.ingest.video_importer import DuplicateVideoError, RightsNotConfirmedError
from clipping_machine.ingest.video_validator import VideoValidationError
from clipping_machine.services import ClippingService
from clipping_machine.storage.models import Video, VideoStatus
from clipping_machine.ui.components.common import (
    Translator,
    format_size,
    format_time,
    issue_messages,
    translator,
)

_STATUS_ICON = {
    VideoStatus.UPLOADED: "⚪",
    VideoStatus.INDEXING: "🔵",
    VideoStatus.INDEXED: "🟢",
    VideoStatus.FAILED: "🔴",
}


def _upload_limit_mb() -> int:
    try:
        return int(st.get_option("server.maxUploadSize"))
    except Exception:
        return 200


def _import_form(service: ClippingService, t: Translator) -> None:
    with st.container(border=True):
        st.subheader(t("library.add"))
        rights = st.checkbox(t("library.rights_checkbox"), key="import_rights")
        extensions = [ext.lstrip(".") for ext in service.settings.upload.allowed_extensions]
        uploaded = st.file_uploader(
            t("library.upload_label"),
            type=extensions,
            accept_multiple_files=False,
            help=t("library.upload_help", limit=_upload_limit_mb()),
            key="import_upload",
        )
        path = st.text_input(
            t("library.path_label"), help=t("library.path_help"), key="import_path"
        )
        left, right = st.columns(2)
        copy = left.checkbox(t("library.path_copy"), value=False, key="import_copy")
        auto_index = right.checkbox(t("library.auto_index"), value=True, key="import_auto")

        if not st.button(t("library.import_button"), type="primary", key="import_submit"):
            return
        if not rights:
            st.warning(t("library.need_rights"))
            return
        if uploaded is None and not path.strip():
            st.warning(t("library.need_file"))
            return
        try:
            if uploaded is not None:
                video = service.import_upload(uploaded, uploaded.name, rights_confirmed=True)
            else:
                video = service.import_path(path, rights_confirmed=True, copy=copy)
        except DuplicateVideoError as exc:
            st.info(t("library.duplicate", name=exc.existing.original_name))
            return
        except VideoValidationError as exc:
            for message in issue_messages(exc.issues):
                st.error(message)
            return
        except (RightsNotConfirmedError, ffmpeg.FFmpegError, OSError) as exc:
            st.error(t("common.error", error=exc))
            return
        if auto_index:
            service.start_indexing(video.video_id)
        st.session_state["library_notice"] = t("library.imported", name=video.original_name)
        st.rerun()


def _video_row(service: ClippingService, video: Video, t: Translator) -> None:
    with st.container(border=True):
        info, actions = st.columns([4, 1])
        with info:
            st.markdown(
                f"**{video.original_name}** &nbsp; {_STATUS_ICON[video.status]} "
                f"{t(f'status.{video.status.value}')}"
            )
            details = [
                format_time(video.duration_ms),
                f"{video.width}×{video.height}",
                f"{video.fps:.0f} fps",
                video.video_codec,
                format_size(video.size_bytes),
            ]
            if video.status is VideoStatus.INDEXED:
                details.append(t("library.segments", count=video.segments_count))
                if video.index_seconds:
                    details.append(t("library.index_time", minutes=video.index_seconds / 60))
            st.caption(" · ".join(details))
            if video.status is VideoStatus.INDEXING:
                st.progress(
                    min(1.0, max(0.0, video.progress)),
                    text=f"{t(f'stage.{video.stage}')} · {video.progress * 100:.0f}%",
                )
            elif video.status is VideoStatus.FAILED:
                st.error(t("library.failed", error=video.error or "?"))
        with actions:
            busy = video.status is VideoStatus.INDEXING
            indexed = video.status is VideoStatus.INDEXED
            label = t("library.reindex") if indexed else t("library.index")
            if st.button(label, key=f"index_{video.video_id}", disabled=busy, width="stretch"):
                service.start_indexing(video.video_id)
                st.session_state["library_notice"] = t("library.queued")
                st.rerun()
            with st.popover(t("library.delete"), disabled=busy, width="stretch"):
                st.write(t("library.delete_confirm"))
                st.caption(
                    t("library.delete_confirm_copy")
                    if video.managed_copy
                    else t("library.delete_confirm_original")
                )
                if st.button(t("library.delete_yes"), key=f"delete_{video.video_id}"):
                    try:
                        service.delete_video(video.video_id)
                    except RuntimeError as exc:
                        st.error(t("common.error", error=exc))
                    else:
                        st.session_state["library_notice"] = t("library.deleted")
                        st.rerun()


def _video_list(service: ClippingService) -> None:
    t = translator()
    videos = service.videos.all()
    if not videos:
        st.info(t("library.empty"))
        return
    st.subheader(t("library.videos"))
    for video in videos:
        _video_row(service, video, t)

    indexing = any(video.status is VideoStatus.INDEXING for video in videos)
    was_polling = st.session_state.get("library_polling", False)
    st.session_state["library_polling"] = indexing
    if was_polling and not indexing:
        # Indexing just finished: refresh the other tabs too.
        st.rerun()


def render(service: ClippingService) -> None:
    t = translator()
    notice = st.session_state.pop("library_notice", None)
    if notice:
        st.success(notice)
    _import_form(service, t)

    indexing = any(v.status is VideoStatus.INDEXING for v in service.videos.all())
    # Poll only while something is being indexed.
    st.fragment(run_every=2.0 if indexing else None)(_video_list)(service)
