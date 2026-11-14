"""Command line interface: ``clipping-machine <command>``.

run             start the local web app
doctor          check FFmpeg, Python packages and the model cache
download-model  fetch the EmbeddingGemma 2 weights once (then you can go offline)
index           import a local file and index it without the UI
search          search indexed videos from the terminal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from clipping_machine import __version__, ffmpeg, privacy


def _format_time(ms: int) -> str:
    total = ms // 1000
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


_WILDCARD_HOSTS = {"0.0.0.0", "::"}  # listen on every interface


def streamlit_command(host: str, port: int) -> list[str]:
    """Command line that starts the web app.

    ``browser.serverAddress`` is always set: when it is missing and the server listens on
    every interface (as inside Docker), Streamlit contacts checkip.amazonaws.com to print
    an "External URL". Setting it keeps the app free of any outbound request.
    """
    app = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    shown = "localhost" if host in _WILDCARD_HOSTS else host
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.serverAddress",
        shown,
        "--browser.gatherUsageStats",
        "false",
    ]


def _cmd_run(args: argparse.Namespace) -> int:
    print(f"Open http://127.0.0.1:{args.port} in your browser (Ctrl+C to stop).")
    return subprocess.call(streamlit_command(args.host, args.port))


def _cmd_doctor(_: argparse.Namespace) -> int:
    from clipping_machine.config import load_settings

    ok = True
    print(f"GTA6 Clipping Machine {__version__}")
    print(f"Python        {sys.version.split()[0]}")
    try:
        print(f"ffmpeg        {ffmpeg.ffmpeg_bin()}")
        print(f"ffprobe       {ffmpeg.ffprobe_bin()}")
    except ffmpeg.FFmpegNotFoundError as exc:
        ok = False
        print(f"ffmpeg        MISSING: {exc}")

    settings = load_settings()
    print(f"data folder   {settings.data_dir}")
    print(f"model         {settings.embedding.model} ({settings.embedding.dimension}d)")
    try:
        import torch

        from clipping_machine.embeddings.embeddinggemma_client import resolve_device

        device = resolve_device(settings.embedding.device)
        print(f"torch         {torch.__version__} on {device}")
        if device == "cpu":
            print("              no GPU detected: indexing works but is slow (see README)")
    except ImportError as exc:
        ok = False
        print(f"torch         MISSING: {exc}")

    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(settings.embedding.model, "config.json")
        local = Path(settings.embedding.model).is_dir()
        state = "cached" if isinstance(cached, str) or local else "not downloaded yet"
        print(f"model weights {state}")
    except Exception as exc:  # diagnostics only
        print(f"model weights unknown ({exc})")
    print(f"offline mode  {'on' if privacy.is_offline() else 'off'}")
    return 0 if ok else 1


def _cmd_download_model(_: argparse.Namespace) -> int:
    from clipping_machine.config import load_settings
    from clipping_machine.embeddings.embeddinggemma_client import EmbeddingGemmaClient

    settings = load_settings()
    print(f"Downloading / verifying {settings.embedding.model} (about 1.5 GB the first time)...")
    client = EmbeddingGemmaClient(settings.embedding)
    client.load()
    vector = client.embed_queries(["test"])[0]
    print(f"OK: model loaded on {client.device}, embedding dimension {vector.shape[0]}.")
    print("You can now set GCM_OFFLINE=1 to block all network access.")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    from clipping_machine.ingest.video_importer import DuplicateVideoError
    from clipping_machine.services import ClippingService

    if not args.i_own_this_footage:
        print(
            "Refusing to import: pass --i-own-this-footage to confirm this is your own "
            "gameplay or that you are authorized to use it.",
            file=sys.stderr,
        )
        return 2
    service = ClippingService.from_environment()
    try:
        video = service.import_path(args.path, rights_confirmed=True, copy=args.copy)
    except DuplicateVideoError as exc:
        video = exc.existing
        print(f"Already in the library as {video.video_id}; re-indexing.")
    print(f"Indexing {video.original_name} ({_format_time(video.duration_ms)}) as {video.video_id}")

    last = {"stage": ""}

    def report(stage: str, overall: float) -> None:
        if stage != last["stage"]:
            last["stage"] = stage
            print(f"  [{overall * 100:5.1f}%] {stage}", flush=True)

    count = service.indexer.index_video(video.video_id, on_progress=report)
    print(f"Done: {count} segments indexed.")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    from clipping_machine.services import ClippingService

    service = ClippingService.from_environment()
    result = service.search(
        args.query,
        video_id=args.video,
        min_similarity=args.min_similarity,
        relative_margin=args.relative_margin,
    )
    names = {video.video_id: video.original_name for video in service.videos.all()}
    print(
        f'"{result.query}": {len(result.moments)} moments '
        f"({result.hits_above_threshold} segments above {result.effective_threshold:.2f}, "
        f"{result.elapsed_ms:.0f} ms)"
    )
    moments = list(result.moments)
    if args.sort == "similarity":
        moments.sort(key=lambda moment: moment.similarity, reverse=True)
    elif args.sort == "time":
        moments.sort(key=lambda moment: (moment.video_id, moment.start_ms))
    for rank, moment in enumerate(moments, start=1):
        flag = " (below threshold)" if moment.below_threshold else ""
        tags = ", ".join(moment.tags) or "-"
        print(
            f"{rank:2d}. {_format_time(moment.start_ms)}-{_format_time(moment.end_ms)}  "
            f"score {moment.score:.2f}  similarity {moment.similarity:.2f}  "
            f"[{tags}]  {names.get(moment.video_id, moment.video_id)}{flag}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipping-machine", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="start the local web app")
    run.add_argument("--host", default="127.0.0.1", help="bind address (default: local only)")
    run.add_argument("--port", type=int, default=8501)
    run.set_defaults(handler=_cmd_run)

    doctor = commands.add_parser("doctor", help="check the installation")
    doctor.set_defaults(handler=_cmd_doctor)

    download = commands.add_parser("download-model", help="download the embedding model")
    download.set_defaults(handler=_cmd_download_model)

    index = commands.add_parser("index", help="import and index a local video file")
    index.add_argument("path", help="path to a video file on this computer (links are rejected)")
    index.add_argument("--copy", action="store_true", help="copy the file into the data folder")
    index.add_argument(
        "--i-own-this-footage",
        action="store_true",
        help="confirm this is your own gameplay or that you are authorized to use it",
    )
    index.set_defaults(handler=_cmd_index)

    search = commands.add_parser("search", help="search indexed videos")
    search.add_argument("query")
    search.add_argument("--video", help="restrict to one video id")
    search.add_argument("--min-similarity", type=float, default=None)
    search.add_argument(
        "--relative-margin",
        type=float,
        default=None,
        help="keep only segments within this distance of the best match (0 = off)",
    )
    search.add_argument(
        "--sort",
        choices=["score", "similarity", "time"],
        default="score",
        help="order of the results (default: editorial score)",
    )
    search.set_defaults(handler=_cmd_search)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    privacy.enforce_local_only()
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
