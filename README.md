# GTA6 Clipping Machine

**Type what happened, get the clip. An AI that watches your gameplay for you, finds the moments
you describe in plain words and turns them into vertical clips for TikTok, Instagram Reels and
YouTube Shorts. Powered by Google's open-source AI model, running entirely on your computer.**

[![CI](https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine/actions/workflows/ci.yml/badge.svg)](https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine/actions/workflows/ci.yml)
[![AI model](https://img.shields.io/badge/AI-Google%20EmbeddingGemma%202-4285F4)](https://huggingface.co/google/embeddinggemma-2)
![Runs locally](https://img.shields.io/badge/runs-100%25%20locally-orange)
![No API key](https://img.shields.io/badge/API%20key-not%20needed-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT%20(free%2C%20with%20credit)-green)](#license-free-to-use-with-credit)

🇧🇷 [Leia em português](README.pt-BR.md)

You recorded three hours of gameplay. Somewhere in there is a police chase that ended in an
explosion, a ridiculous fall and a perfect drift. Finding them means scrubbing through the whole
file. Instead, type **"police chase with explosion"** and get a ranked list of moments with a
timeline, previews and a score. Trim one, approve it and export a ready-to-post 1080x1920 MP4.

```text
Query: police chase with explosion                          (illustrative output)

1. 12:04–12:19   score 0.81   police chase, explosion · loud audio · high motion
2. 27:41–27:58   score 0.74   vehicle on fire, high-speed driving · scene change
3. 35:10–35:26   score 0.70   motorcycle, police chase
```

- **No upload.** Your footage never leaves your computer.
- **No subscription, no API key, no per-minute cost.** The AI runs on your own machine, free.
- **No GPU required.** It runs on an ordinary laptop CPU; an NVIDIA GPU just makes it faster.
- **Search in your language.** The model understands 100+ languages: "perseguição policial com
  explosão" finds the same moment as the English query.
- **Open source end to end.** This app is MIT-licensed; the AI model is Apache-2.0.

> Built for GTA, works with any game. Not affiliated with Rockstar Games or Take-Two Interactive.

## The AI behind it: Google's open-source EmbeddingGemma 2

The search engine is [**EmbeddingGemma 2**](https://huggingface.co/google/embeddinggemma-2), an
open-source multimodal AI model released by Google under the Apache-2.0 license, from the same
family as Google's Gemma models. It has about 744 million parameters, small enough to run on a
laptop, and it is one of the few open models that understands **text, images, video and audio
in a single shared "meaning space"**.

What that means in practice:

1. Every 8-second window of your recording (four frames plus its audio) is turned into a list
   of 768 numbers, an *embedding*, that captures what is happening: a car chase at night, a
   rooftop party, a gunfight in an alley.
2. Your query is turned into 768 numbers in the same space.
3. The windows whose numbers are closest to your query are the moments you are looking for. No
   keywords, no manual tagging, no timestamps to remember: the AI matches **meaning**.

The model is downloaded once from Hugging Face (about 1.5 GB) and then runs offline on your
machine. It is used only to *understand* your footage; it does not generate content, and nothing
is sent to Google or anyone else.

| | Typical cloud clipping service | GTA6 Clipping Machine |
|---|---|---|
| Your footage | uploaded to someone's server | stays on your disk |
| Cost | monthly plan or per-minute credits | free |
| Account / API key | required | none |
| Works offline | no | yes, after the one-time model download |
| Source code | closed | open (MIT), every step inspectable |
| Publishing | some can post to your accounts for you | never: you approve and post it yourself |

## Features

- **Natural-language search** over your recordings: "drift on a tight corner", "funny fall",
  "intense shootout and escape", "cinematic sunset".
- **Moments, not frames**: nearby matches are merged into clip-sized intervals (8–60 s,
  aiming for 15–45 s).
- **Editorial score** that combines semantic match, audio intensity, motion, reaction presence,
  novelty compared with clips you already made, and duration fit. Every score comes with a
  breakdown and a plain-language reason.
- **Timeline** of the moments over the whole video, with audio and motion curves.
- **Manual trim** of start and end, with a live re-score.
- **Vertical export**: 1080x1920 H.264/AAC MP4 with *smart crop* (follows the action),
  centre crop, or the full frame over a blurred background. Black bars recorded around the
  picture (4:3 games, letterboxed cutscenes, OBS canvases) are removed automatically.
  Automatic thumbnail. Optional burned-in subtitles, transcribed locally.
- **Approval queue**: every clip waits for your decision. Nothing is published by the app.
- **Private by design**: local files only (no link downloading), local processing, no
  telemetry, works fully offline after the first model download.
- English and Brazilian Portuguese interface. A CLI for scripting.

## How it works

```text
your video ─► validate ─► scenes ─► keyframes every 2 s ─┐
                                    audio (16 kHz mono) ──┼─► EmbeddingGemma 2 ─► LanceDB index
                                    motion, loudness ─────┘   (8 s windows,        (local)
                                                               2 s overlap)
"your query" ─► EmbeddingGemma 2 ─► nearest windows ─► merge into moments ─► score ─► timeline
                                                                                        │
                       approval queue ◄── generate clip ◄── trim start/end ◄────────────┘
                             │
                             └─► approve ─► export 9:16 MP4 (FFmpeg)
```

Under the hood, everything is open source and local: **EmbeddingGemma 2** (understanding),
**FFmpeg** (decoding, cropping and encoding), **PySceneDetect** (scene cuts), **LanceDB**
(vector search), **SQLite** (state and history) and **Streamlit** (the web interface on
`127.0.0.1`).

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · model setup and measurements:
[docs/MODELS.md](docs/MODELS.md).

## Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10/11, macOS, Linux | |
| Python | 3.11 | 3.12 or 3.13 |
| FFmpeg | any recent build on `PATH` | 6.0+ |
| RAM | 16 GB | 32 GB |
| GPU | not required (CPU works) | NVIDIA GPU with 8 GB+ VRAM |
| Disk | ~5 GB (model + dependencies) plus your recordings | SSD |

On CPU, indexing takes roughly **1–1.5× the length of the video**, once per recording
(measured on an 8-core Ryzen 7 laptop without a dedicated GPU: a 9-minute 1080p gameplay
recording in 11.5 minutes). After that, every search takes **about 0.3 seconds**, as many times
as you like. A CUDA GPU indexes much faster and is used automatically.

## Quick start

### Easiest: let your AI coding assistant install it

Using **Claude Code**, **OpenAI Codex**, Cursor or another AI coding agent? Paste this:

```text
Install and start https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine on my
computer, following the AGENTS.md file in that repository.
```

The agent checks Python and FFmpeg, installs the app, downloads the AI model and opens it for
you. The step-by-step instructions it follows are in [AGENTS.md](AGENTS.md). Prefer to do it
yourself? Read on.

### 1. Install FFmpeg

```bash
winget install Gyan.FFmpeg        # Windows
brew install ffmpeg               # macOS
sudo apt install ffmpeg           # Debian / Ubuntu
```

### 2. Install the app

```bash
git clone https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine.git
cd GTA6-Clipping-Machine
python -m venv .venv
```

Activate the environment:

```bash
.venv\Scripts\activate            # Windows (PowerShell or cmd)
source .venv/bin/activate         # macOS / Linux
```

Optional, for an NVIDIA GPU: install the CUDA build of PyTorch first, using the command for
your system from [pytorch.org](https://pytorch.org/get-started/locally/).

```bash
pip install -e .
```

### 3. Check the setup and download the model (once, ~1.5 GB)

```bash
clipping-machine doctor
clipping-machine download-model
```

After this you can work fully offline: set `GCM_OFFLINE=1` (see [Configuration](#configuration)).

### 4. Start the app

```bash
clipping-machine run
```

Open <http://127.0.0.1:8501>. The app only listens on your own computer.

### Docker (optional)

```bash
docker compose up --build app          # CPU → http://127.0.0.1:8501
```

Put recordings in `./videos` and import them by path (`/videos/my-session.mp4`) so they are
indexed without copying. Data and model weights are kept in `./data` and `./models`. For an
NVIDIA GPU, see the comments in [docker-compose.yml](docker-compose.yml).

## Using the app

1. **Videos**: confirm that the footage is yours, then drag and drop a file (MP4, MOV, MKV,
   WEBM; up to 4 GB per upload) or type the path of a file on your disk. Importing by path
   indexes huge recordings **in place**, without copying. Indexing starts automatically and
   shows its progress.
2. **Find moments**: describe what you are looking for. Adjust the minimum similarity, the
   number of segments and the merge gap under *Search settings*. Filter by score and duration.
   Click a bar on the timeline to jump to that moment.
3. On a moment: check the **score**, the **reason** and the **breakdown**, watch a
   **preview**, drag the **start/end** slider, then **Generate clip**.
4. **Approval queue**: watch the review draft, adjust it, **approve** or **reject**. Approved
   clips can be **exported** with the framing of your choice (and subtitles if installed).
   Download the MP4 or find it in `data/clips/`. When you have posted it yourself, you can mark
   it as published.
5. **Status & privacy**: model and FFmpeg status, local metrics (searches, average search
   time, approval rate, clips exported, indexing speed).

Exported files are named `gta6_clip_<date>_<position in the video>_score<NNN>.mp4`, for
example `gta6_clip_2026-11-14_00-12-04_score081.mp4`.

### Command line

```bash
clipping-machine index "D:\Recordings\session-01.mkv" --i-own-this-footage
clipping-machine search "drift on a tight corner"
clipping-machine search "explosão" --sort similarity --min-similarity 0.6
```

## Configuration

All settings live in [`configs/`](configs/) and are validated at start-up.

| File | What it controls |
|---|---|
| [`processing.yaml`](configs/processing.yaml) | upload limits, window length and overlap, keyframe interval, scene detection, search threshold and merging, score weights, export size and quality |
| [`models.yaml`](configs/models.yaml) | EmbeddingGemma 2 (device, precision, dimension, vision tokens, audio on/off), subtitles model |
| [`tags.yaml`](configs/tags.yaml) | the concepts used for the "reason" tags and the reaction cues; describe *your* gameplay, in any language |

Defaults follow the project spec: 8 s windows, 2 s overlap, a keyframe every 2 s, 768-d
embeddings, minimum similarity 0.65, merge gap 3 s, target clip length 15–45 s, maximum 60 s.

Environment variables (copy [`.env.example`](.env.example) to `.env`):

| Variable | Purpose |
|---|---|
| `GCM_OFFLINE=1` | never contact the network (use after `download-model`) |
| `GCM_DEVICE` | `auto`, `cuda`, `mps` or `cpu` |
| `GCM_EMBEDDING_MODEL` | Hub id or a local folder with the model |
| `GCM_DATA_DIR` | where videos, index, previews and clips are stored |
| `GCM_LANG` | interface language, `en` or `pt` |

Subtitles need an extra: `pip install -e ".[subtitles]"`.

## Privacy

- Video, audio, frames and embeddings **never leave your computer**.
- The only network access is downloading the model weights the first time. `GCM_OFFLINE=1`
  blocks even that.
- No telemetry. Hugging Face and Streamlit usage statistics are switched off. The metrics on
  the Status page are stored in a local SQLite file.
- The web app listens on `127.0.0.1` only.
- You can delete a video and everything derived from it (index, previews, clips) from the app.

## Acceptable use

Use this tool **only with gameplay you recorded yourself** from a legitimate copy of the game,
or footage you are authorized to use. It does not download videos from links, by design, and
it never publishes anything for you. You are responsible for the rights to what you import and
post, and for following the game publisher's content policy and each platform's rules. Full
text: [docs/ACCEPTABLE_USE.md](docs/ACCEPTABLE_USE.md).

The score helps you review the most promising moments first. It does **not** predict views or
virality.

## Limitations

- **Semantic search is not event detection.** The model matches the overall look and sound of
  an 8-second window. Very specific or brief events can be missed; rephrasing the query, or
  lowering the minimum similarity, often helps.
- **Similarities are close together.** With this model, unrelated content scores around 0.5–0.6
  and clear matches around 0.7–0.78. Results must clear 0.65 *and* be within 0.04 of the
  query's best match (the *relevance window* under Search settings); widen the window if a
  search returns too little.
- **Loud moments get a boost.** Audio intensity is 20% of the score. The relevance window keeps
  a loosely related loud scene from jumping ahead of the real match, but among moments of
  similar relevance the louder one usually ranks first. Change the weights in
  `processing.yaml`, or sort by similarity in the CLI.
- **Tags are hints**, computed by comparing each window with short descriptions in
  `tags.yaml`. They can be wrong.
- **Reaction presence is a heuristic** based on similarity to descriptions such as "a streamer
  reacting". It works best with a visible face camera or loud voice reactions.
- **Smart crop** follows horizontal action with one crop position per clip, and only a little:
  the centre of the screen, where third-person games keep the player or the car, always stays
  in frame. It does not track a subject that moves across the frame. Over-the-shoulder aiming
  puts the character left of centre, outside a 9:16 crop; use the blurred-background framing
  for those clips.
- **CPU indexing is slow** for long sessions (about real time or slower). A GPU helps a lot.
- Tested by the maintainer on Windows 11 (CPU). Linux and Windows run the test suite in CI;
  GPU and macOS support follow the libraries' standard paths but have had less testing.
  Reports are welcome.

## Development

```bash
pip install -e ".[dev]"
python -m pytest              # 193 tests; no model download (a fake embedder is used)
python -m pytest -m model     # opt-in test with the real EmbeddingGemma 2
python -m ruff check src tests && python -m ruff format --check src tests
python -m mypy                # strict
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [roadmap](docs/ROADMAP.md).

## License: free to use, with credit

GTA6 Clipping Machine is free software under the [MIT License](LICENSE), created by
**Rodrigo Florencio** ([@RodrigoFlorencio86](https://github.com/RodrigoFlorencio86)).

- ✅ **Use it for free**, personally or commercially: for your own channel, for clients, in a
  studio.
- ✅ **Modify it, fork it, build on it, redistribute it**, including inside other products.
- 📝 **The one condition: credit the author.** Any copy, fork or product that includes this
  code, or a substantial part of it, must keep the copyright notice
  `Copyright (c) 2026 Rodrigo Florencio` and the license text.
- 🎬 **The clips you make are yours.** No watermark, and no credit is required inside your
  videos. If the tool saves you time, a mention is always appreciated:

  ```text
  Clipped with GTA6 Clipping Machine by Rodrigo Florencio
  https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine
  ```

EmbeddingGemma 2 is distributed by Google under the Apache-2.0 license; FFmpeg, PyTorch, LanceDB,
Streamlit and PySceneDetect are covered by their own licenses.

If this project is useful to you, **give it a ⭐ on GitHub**: it helps other creators find it.

*Grand Theft Auto and GTA are trademarks of Take-Two Interactive Software, Inc. This project is
independent and is not affiliated with, endorsed by or sponsored by Rockstar Games, Take-Two
Interactive, Google, TikTok, Instagram or YouTube.*
