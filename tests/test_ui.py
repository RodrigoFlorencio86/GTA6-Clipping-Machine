"""Streamlit smoke tests with ``AppTest`` (real UI code, fake embedding model)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from clipping_machine.config import Settings
from clipping_machine.services import ClippingService
from clipping_machine.storage.models import ClipStatus, VideoStatus
from tests.fakes import FakeEmbedder

APP = Path(__file__).resolve().parents[1] / "src" / "clipping_machine" / "ui" / "streamlit_app.py"


@pytest.fixture
def ui_service(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> Iterator[ClippingService]:
    embedder = FakeEmbedder(window_concepts={1: "chase"})
    service = ClippingService(settings, backend=embedder)
    monkeypatch.setattr(ClippingService, "from_environment", classmethod(lambda cls: service))
    st.cache_resource.clear()
    yield service
    st.cache_resource.clear()
    service.jobs.shutdown()


def _app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    return app


def test_app_renders_every_tab(ui_service: ClippingService) -> None:
    app = _app()
    assert app.title[0].value == "GTA6 Clipping Machine"
    assert [tab.label for tab in app.tabs] == [
        "Videos",
        "Find moments",
        "Approval queue",
        "Status & privacy",
    ]
    assert any("Index a video first" in info.value for info in app.info)


def test_language_switch_to_portuguese(ui_service: ClippingService) -> None:
    app = _app()
    app.selectbox(key="language").select("pt").run()
    assert not app.exception
    assert app.tabs[0].label == "Vídeos"


def test_import_requires_the_rights_checkbox(
    ui_service: ClippingService, sample_video: Path
) -> None:
    app = _app()
    app.text_input(key="import_path").input(str(sample_video))
    app.button(key="import_submit").click().run()
    assert not app.exception
    assert any("ownership" in warning.value for warning in app.warning)
    assert ui_service.videos.all() == []


def test_links_are_refused_in_the_ui(ui_service: ClippingService) -> None:
    app = _app()
    app.checkbox(key="import_rights").check()
    app.text_input(key="import_path").input("https://www.youtube.com/watch?v=abc")
    app.button(key="import_submit").click().run()
    assert any("Links are not supported" in error.value for error in app.error)
    assert ui_service.videos.all() == []


@pytest.mark.ffmpeg
def test_full_flow_from_import_to_export(ui_service: ClippingService, sample_video: Path) -> None:
    app = _app()

    # 1. Import a local file (indexed in place) without auto-indexing, then index it.
    app.checkbox(key="import_rights").check()
    app.checkbox(key="import_auto").uncheck()
    app.text_input(key="import_path").input(str(sample_video))
    app.button(key="import_submit").click().run()
    assert not app.exception
    [video] = ui_service.videos.all()
    ui_service.index_now(video.video_id)
    assert ui_service.videos.require(video.video_id).status is VideoStatus.INDEXED

    # 2. Search.
    app.run()
    app.text_input(key="search_query").input("police chase")
    app.checkbox(key="search_fill").uncheck()
    next(b for b in app.button if b.label == "Search").click().run()
    assert not app.exception, app.exception
    assert any("1 moments" in caption.value for caption in app.caption)

    # 3. Generate a clip from the first moment (renders a review draft).
    next(b for b in app.button if b.label == "Generate clip").click().run()
    assert not app.exception, app.exception
    [clip] = ui_service.clips.find()
    assert clip.status is ClipStatus.CANDIDATE
    assert clip.draft_path and Path(clip.draft_path).is_file()
    assert clip.reason

    # 4. Approve it in the queue, then export.
    app.button(key=f"approve_{clip.clip_id}").click().run()
    assert not app.exception, app.exception
    assert ui_service.clips.require(clip.clip_id).status is ClipStatus.APPROVED
    app.selectbox(key=f"framing_{clip.clip_id}").select("center")
    app.button(key=f"export_{clip.clip_id}").click().run()
    assert not app.exception, app.exception
    exported = ui_service.clips.require(clip.clip_id)
    assert exported.status is ClipStatus.EXPORTED
    assert exported.output_path and Path(exported.output_path).is_file()

    # 5. The status page renders the local metrics.
    assert any(metric.label == "Clips exported" for metric in app.metric)
