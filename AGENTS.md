# Instructions for AI coding agents

You are probably here because a user gave you this repository's link and asked you to install
GTA6 Clipping Machine on their computer. This file tells you how. (If you are instead working
on the code itself, read [CONTRIBUTING.md](CONTRIBUTING.md) and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); `CLAUDE.md` holds the maintainer's notes.)

## Goal

Leave the user with the web app running at <http://127.0.0.1:8501> and the AI model already
downloaded, then tell them how to start it again later.

## Steps

Work in a terminal on the user's machine. Explain briefly what you are doing; the user may not
be a developer. Ask before installing system software (Python, FFmpeg, Git) if your environment
requires confirmation.

1. **Python 3.11 or newer.** Check with `python --version` (or `python3 --version`). If it is
   missing or older:
   - Windows: `winget install Python.Python.3.12`
   - macOS: `brew install python@3.12`
   - Debian/Ubuntu: `sudo apt install python3 python3-venv python3-pip`
2. **FFmpeg** on the `PATH`. Check with `ffmpeg -version`. If it is missing:
   - Windows: `winget install Gyan.FFmpeg` (open a new terminal afterwards so `PATH` updates)
   - macOS: `brew install ffmpeg`
   - Debian/Ubuntu: `sudo apt install ffmpeg`
3. **Get the code**, into a folder the user can find again (for example their home folder):

   ```bash
   git clone https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine.git
   cd GTA6-Clipping-Machine
   ```

   No Git? Download <https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine/archive/refs/heads/main.zip>
   and extract it.
4. **Virtual environment and install:**

   ```bash
   python -m venv .venv
   # Windows:      .venv\Scripts\activate
   # macOS/Linux:  source .venv/bin/activate
   pip install -e .
   ```

   If the computer has an NVIDIA GPU (`nvidia-smi` works), install the CUDA build of PyTorch
   **before** `pip install -e .`, using the command for that system from
   <https://pytorch.org/get-started/locally/>. Otherwise skip this: the CPU build works.
   The install downloads a few GB of Python packages; it can take several minutes.
5. **Check and download the model** (about 1.5 GB, once; no account or token needed):

   ```bash
   clipping-machine doctor
   clipping-machine download-model
   ```

   `doctor` must show a path for `ffmpeg` and `ffprobe`, a `torch` version and, after the
   download, `model weights cached`. "No GPU detected" is only a speed note, not an error.
   Fix anything else it flags before going on.
6. **Start the app** and leave it running (in the background if your tool allows it):

   ```bash
   clipping-machine run
   ```

   When the log shows `URL: http://127.0.0.1:8501`, give the user that link.
7. **Tell the user** how to start it next time: open a terminal in the project folder, activate
   the environment (step 4) and run `clipping-machine run`. Mention that after the first model
   download they can set `GCM_OFFLINE=1` to keep the app fully offline, and that the interface
   is available in English and Brazilian Portuguese (language selector in the sidebar).

## Rules

- Keep it local. Do not start the app with `--host 0.0.0.0` or expose it to the network unless
  the user explicitly asks; do not deploy it to any cloud service.
- Do not upload the user's videos anywhere, and do not download videos from YouTube, TikTok,
  Twitch or other sites for them: the app only works with files on their own disk, by design.
- Do not edit the code or the files in `configs/` unless the user asks.
- If something fails, read the error, fix the cause (missing FFmpeg, old Python, no disk
  space) and retry. On machines with 16 GB of RAM or less, close other heavy programs before
  indexing.

## Quick checks

| Symptom | Fix |
|---|---|
| `ffmpeg` not found after installing it on Windows | open a new terminal, or restart the editor |
| `clipping-machine` not found | the virtual environment is not active (step 4) |
| model download fails | check the internet connection and that `GCM_OFFLINE` is not set |
| indexing is slow | expected on CPU: about 1–1.5× the video length, once per recording |
