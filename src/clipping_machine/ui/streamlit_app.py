"""Streamlit entry point.

Run with ``clipping-machine run`` or ``streamlit run src/clipping_machine/ui/streamlit_app.py``.
"""

from __future__ import annotations

from clipping_machine import privacy

privacy.enforce_local_only()  # before any library that could phone home is imported

import os  # noqa: E402

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from clipping_machine import ffmpeg  # noqa: E402
from clipping_machine.services import ClippingService  # noqa: E402
from clipping_machine.ui import i18n  # noqa: E402
from clipping_machine.ui.components import about, library, queue, search  # noqa: E402
from clipping_machine.ui.components.common import translator  # noqa: E402


@st.cache_resource(show_spinner=False)
def get_service() -> ClippingService:
    """One service (and one loaded model) shared by every browser session."""
    load_dotenv()
    privacy.enforce_local_only()
    return ClippingService.from_environment()


def _sidebar() -> None:
    with st.sidebar:
        if "language" not in st.session_state:
            preferred = os.environ.get("GCM_LANG", i18n.DEFAULT_LANGUAGE).lower()[:2]
            st.session_state["language"] = (
                preferred if preferred in i18n.LANGUAGES else i18n.DEFAULT_LANGUAGE
            )
        st.selectbox(
            i18n.translate(st.session_state["language"], "sidebar.language"),
            options=list(i18n.LANGUAGES),
            format_func=lambda code: i18n.LANGUAGES[code],
            key="language",
        )
        t = translator()
        st.success(f"**{t('sidebar.local')}**\n\n{t('sidebar.local_body')}", icon="🔒")
        st.caption(t("sidebar.rights"))
        if not ffmpeg.is_available():
            st.error(t("sidebar.ffmpeg_missing"))


def main() -> None:
    st.set_page_config(page_title="GTA6 Clipping Machine", page_icon="🎬", layout="wide")
    service = get_service()
    _sidebar()
    t = translator()
    st.title("GTA6 Clipping Machine")
    st.caption(t("app.tagline"))

    tabs = st.tabs([t("tab.library"), t("tab.search"), t("tab.queue"), t("tab.about")])
    with tabs[0]:
        library.render(service)
    with tabs[1]:
        search.render(service)
    with tabs[2]:
        queue.render(service)
    with tabs[3]:
        about.render(service)


main()
