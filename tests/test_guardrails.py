"""Project rules enforced as tests: local-only, no downloaders, no auto-publishing."""

from __future__ import annotations

import ast
import os
import tomllib
from pathlib import Path

from clipping_machine import privacy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "clipping_machine"

# Modules that would let the app fetch remote media, call web APIs or post content.
FORBIDDEN_IMPORTS = {
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    "yt_dlp",
    "youtube_dl",
    "pytube",
    "socket",
    "ftplib",
    "smtplib",
    "boto3",
    "botocore",
    "googleapiclient",
    "google",
    "tweepy",
    "TikTokApi",
    "instagrapi",
    "paramiko",
    "websocket",
    "websockets",
    "posthog",
    "sentry_sdk",
    "segment",
}
FORBIDDEN_FROM = {"urllib.request", "http.client"}
FORBIDDEN_PACKAGES = {
    "yt-dlp",
    "youtube-dl",
    "pytube",
    "requests",
    "httpx",
    "boto3",
    "google-api-python-client",
    "tweepy",
    "posthog",
    "sentry-sdk",
}


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_source_has_no_network_or_publishing_imports() -> None:
    offenders: list[str] = []
    for path in SOURCE.rglob("*.py"):
        for module in _imports(path):
            if module.split(".")[0] in FORBIDDEN_IMPORTS or module in FORBIDDEN_FROM:
                offenders.append(f"{path.relative_to(ROOT)}: {module}")
    assert offenders == []


def test_dependencies_contain_no_downloaders_or_uploaders() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    declared = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        declared.extend(extra)
    names = {
        dep.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip().lower()
        for dep in declared
    }
    assert names.isdisjoint(FORBIDDEN_PACKAGES)


def test_there_is_no_publisher_module() -> None:
    publishing = {path.name for path in (SOURCE / "publishing").glob("*.py")}
    assert publishing == {"__init__.py", "approval_queue.py"}
    names = {path.stem.lower() for path in SOURCE.rglob("*.py")}
    assert not any(
        "publisher" in name or "uploader" in name or "download" in name for name in names
    )


def test_telemetry_is_disabled() -> None:
    env: dict[str, str] = {}
    privacy.enforce_local_only(env)
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"
    assert env["DO_NOT_TRACK"] == "1"
    assert "HF_HUB_OFFLINE" not in env


def test_offline_mode_blocks_model_downloads() -> None:
    env = {"GCM_OFFLINE": "1"}
    privacy.enforce_local_only(env)
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert privacy.is_offline(env) and not privacy.is_offline({})


def test_streamlit_is_local_only_and_does_not_report_usage() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["address"] == "127.0.0.1"
    assert config["client"]["toolbarMode"] == "minimal"  # no "Deploy to the cloud" button


def test_web_app_never_looks_up_the_external_ip() -> None:
    """Streamlit calls checkip.amazonaws.com unless browser.serverAddress is set."""
    from clipping_machine.cli import streamlit_command

    for host in ("127.0.0.1", "0.0.0.0", "::"):
        command = streamlit_command(host, 8501)
        option = command.index("--browser.serverAddress")
        assert command[option + 1] in {"127.0.0.1", "localhost"}
        assert command[command.index("--browser.gatherUsageStats") + 1] == "false"
        assert command[command.index("--client.toolbarMode") + 1] == "minimal"


def test_importing_the_app_disables_telemetry_in_the_process() -> None:
    import clipping_machine.services  # noqa: F401

    privacy.enforce_local_only()
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_no_secrets_are_committed() -> None:
    assert not (ROOT / ".env").exists() or ".env" in (ROOT / ".gitignore").read_text()
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "API_KEY" not in example.upper().replace("NO API_KEY", "")
