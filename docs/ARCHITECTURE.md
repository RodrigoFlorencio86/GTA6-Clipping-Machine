# Architecture

GTA6 Clipping Machine is a single local Python process: a Streamlit UI (or the CLI) on top of
one service object that owns the database, the vector index, the embedding model and FFmpeg.
Nothing runs on a server and nothing leaves the machine.

```text
          ┌───────────── Streamlit UI (ui/) ─────────────┐   ┌──── CLI (cli.py) ────┐
          │ Videos · Find moments · Approval queue · Status│   │ index · search · ... │
          └───────────────────────┬────────────────────────┘   └──────────┬───────────┘
                                  └────────────► ClippingService ◄─────────┘
                                                 (services.py)
     ┌──────────────┬──────────────┬─────────────┼──────────────┬──────────────┬─────────────┐
  ingest/       pipeline/      embeddings/     search/        clipping/     publishing/   storage/
  validate      indexer +      EmbeddingGemma  LanceDB index  9:16 render   approval      SQLite
  import        job thread     2 client        merge + score  thumbnails    state machine repos
  scenes                                                       subtitles
  keyframes                         │               │              │
  audio ─── FFmpeg ─────────────────┘               │              └─── FFmpeg
                                    data/index/ (LanceDB)   data/clipping_machine.sqlite3
```

## Data on disk

Everything lives under `data/` (or `GCM_DATA_DIR`):

| Path | Contents |
|---|---|
| `uploads/` | Copies of files uploaded through the browser (files imported by path stay where they are) |
| `previews/<video_id>/kf_000123.jpg` | Keyframes, one every 2 s; used as model input and as previews |
| `previews/<video_id>/moments/` | Low-resolution previews of search results |
| `processed/` | Scratch space (the 16 kHz WAV during indexing); emptied afterwards |
| `index/` | LanceDB tables, one per embedding dimension (`segments_d768`) |
| `clips/<video_id>/` | Review drafts and final MP4 + JPEG thumbnail |
| `clipping_machine.sqlite3` | Videos, scenes, segments, clips, clip history, local search metrics |

Deleting a video removes its index rows, previews, drafts, exports and database rows. The
original file is deleted only if the app made its own copy of it.

## Indexing pipeline

`pipeline/indexer.py` runs these stages in a background thread (one video at a time). Progress
is written to SQLite, so every browser tab can show it and a crash leaves a visible `failed`
state rather than a stuck one.

| # | Stage | Module | Notes |
|---|---|---|---|
| 1–3 | Upload, validation, metadata | `ingest/video_importer.py`, `video_validator.py`, `ffmpeg.probe` | Extension, size, duration, resolution and a real ffprobe read. URLs are rejected. Ownership must be confirmed. A content fingerprint blocks duplicates. |
| 4 | Scene detection | `ingest/scene_splitter.py` | PySceneDetect `ContentDetector`. Cuts feed the "scene change" reason and are stored for future boundary snapping. |
| 5 | Keyframes + motion | `ingest/frame_extractor.py` | One FFmpeg decode produces a JPEG every 2 s **and** a 4 fps grayscale stream for motion (mean absolute frame difference). |
| 6 | Audio | `ingest/audio_chunker.py` | Mono 16 kHz PCM WAV; read per window. |
| 7 | Embeddings | `embeddings/` | Windows of 8 s, stride 6 s (2 s overlap). Each window = its 4 keyframes as a short video + its 8 s of audio, embedded together into one 768-d vector. |
| 8 | Storage | `search/vector_store.py`, `storage/` | Vectors → LanceDB (cosine). Per-segment signals (loudness, motion, cuts, tags, reaction) → SQLite. |

Per-video normalization: audio loudness and motion are scaled against the video's own 5th–95th
percentiles, so a quiet recording and a loud one rank comparably.

Zero-shot tags (`analysis/scene_describer.py`): each concept in `configs/tags.yaml` is embedded
once as a text query. A segment gets a tag when its similarity stands out within the video
(z-score ≥ 1) **and** clears an absolute cosine floor (0.65). Without the floor, footage with
none of the concepts still received "relative winner" tags.

## Search

1. **Query embedding** with the model's `SearchQuery` task prompt.
2. **Vector search** (`top_k`, default 50) with cosine similarity, optionally restricted to one video.
3. **Threshold**: a hit counts only if its raw cosine is at least
   `max(min_similarity, best_similarity − relative_margin)`, i.e. at least 0.65 **and** within
   0.04 of the query's best match. The relative part exists because similarities sit in a
   narrow band: on a real 9-minute recording, 50 of 88 windows cleared 0.65 for a generic
   query, so loosely related but loud scenes outranked the right one and neighbouring hits
   merged into 40–56 s blocks of mixed content. `relative_margin: 0` turns it off.
4. **Interval merging** (`search/interval_merger.py`):
   - hits whose gap is < 3 s are merged into one interval;
   - intervals longer than 60 s are split into equal parts no longer than 45 s;
   - intervals shorter than 8 s are expanded around their centre, inside the video.
5. **Editorial score** (`analysis/highlight_scorer.py`):

   ```text
   score = 0.35 * semantic_similarity
         + 0.20 * audio_intensity
         + 0.15 * visual_motion
         + 0.15 * reaction_presence
         + 0.10 * novelty
         + 0.05 * duration_fit
   ```

   | Signal | Computation |
   |---|---|
   | semantic_similarity | Interval similarity (0.7·max + 0.3·mean of its hits), rescaled from the cosine band [0.55, 0.80] to [0, 1] |
   | audio_intensity | 0.5·max + 0.5·mean of the segments' normalized loudness (mean and 95th-percentile short-time RMS in dBFS) |
   | visual_motion | Same aggregate over normalized motion |
   | reaction_presence | Max over segments of the similarity to reaction cues ("a streamer reacting…"), mapped from [0.62, 0.76] to [0, 1] |
   | novelty | 1 − max(temporal overlap with an existing clip of the same video, rescaled embedding similarity to any existing clip) |
   | duration_fit | 1 inside 15–45 s, linear down to 0.5 at 8 s and to 0 at 60 s |

   Weights live in `configs/processing.yaml` and must sum to 1.
6. **Ranking and filling**: moments are sorted by score. Because they all passed step 3, the
   score only reorders moments of similar relevance. If fewer than 10 pass, the best
   remaining moments (by similarity) are appended and clearly marked, so the user always has
   candidates to review without the threshold silently changing.

A manual start/end change re-runs step 5 for the new interval.

## Clips and approval

`publishing/approval_queue.py` is a state machine:

```text
candidate ──approve──► approved ──export──► exported ──user posted it──► published ──► archived
    │  ▲                  │                    │
    │  └──── reopen ──────┘◄── export again ───┘
    └──reject──► rejected ──► archived
```

- Creating a clip from a moment stores the interval, query, reason, score and breakdown, and
  renders a 360x640 review draft.
- `export_clip` refuses anything not in `approved`. It renders 1080x1920 H.264/AAC MP4 with
  `+faststart`, strips source metadata, writes a thumbnail and never overwrites an earlier export.
- Framing (`clipping/vertical_formatter.py`):
  - Black borders recorded around the picture (a 4:3 game in a 16:9 capture, letterboxed
    cutscenes, an OBS canvas larger than the game) are cut first. One low-resolution pass
    (320 px wide, 4 fps) over the clip marks a row or column as border when its mean
    brightness stays at or below `black_level` (24) in every sampled frame, like FFmpeg's
    `cropdetect`. Borders under 1% of the frame are ignored, and a box under half the frame
    is rejected, so full-frame footage is left untouched. The same pass feeds the smart crop.
  - `smart`: scale to fill and crop, shifted towards localized motion. Whole-frame change
    (camera pans) is subtracted first, so only a moving subject pulls the crop.
  - `center`: plain centre crop.
  - `blur_pad`: the full frame over a blurred, zoomed copy of itself.
- Subtitles (optional, `[subtitles]` extra): the clip's audio is transcribed locally with
  faster-whisper, written as SRT and burned in.

## Design decisions

- **One service, many front-ends.** The UI and the CLI only call `ClippingService`, so features
  are testable without Streamlit and a future local API (e.g. FastAPI) is a thin layer.
- **Backend protocol.** Everything depends on `embeddings.base.EmbeddingBackend`. Tests use a
  deterministic fake; another local model can be plugged in without touching the pipeline.
- **Pre-decoded frames.** The model gets frames that FFmpeg already decoded. That removes any
  dependency on torch video decoders, keeps keyframe extraction to one pass, and lets the same
  JPEGs serve as previews.
- **Cosine stays cosine.** The 0.65 threshold is compared with raw cosine similarity; only the
  score uses a calibrated value. The number on the slider means what it says.
- **Index per model.** Rows carry the embedding model name and tables are per dimension, so
  changing the model or the Matryoshka size never mixes incomparable vectors.
- **No hidden state.** SQLite holds the pipeline state, so a restart reports interrupted jobs
  instead of leaving videos stuck in "indexing".

## Differences from the original spec layout

| Spec | Here | Why |
|---|---|---|
| `src/<package>/` folders at the top of `src/` | `src/clipping_machine/...` | Installable package with a `clipping-machine` command |
| `embeddings/text_embedder.py`, `image_embedder.py`, `audio_embedder.py` | Methods of `EmbeddingGemmaClient` | One model object serves every modality |
| `publishing/youtube_publisher.py` | Not present | The project does not publish automatically |
| `configs/channels.yaml`, `prompts/` | Not present yet | Only useful with multi-channel presets and a local VLM (roadmap) |
| FastAPI endpoints (spec §10) | `ClippingService` methods with the same inputs/outputs | The MVP UI is Streamlit; a local API can wrap the service later |
