# Roadmap

The phases follow the original product spec. Items marked ✅ ship in v0.1.0.

## Phase 1 — Search MVP ✅

- ✅ Local upload (drag-and-drop) and in-place indexing of large local files
- ✅ Validation: format, size, duration, resolution, readable streams; links rejected
- ✅ Scene detection (PySceneDetect), keyframes every 2 s, 16 kHz mono audio chunks
- ✅ EmbeddingGemma 2 embeddings for 8 s windows with 2 s overlap (frames + audio together)
- ✅ Local vector index (LanceDB) and metadata/pipeline state (SQLite)
- ✅ Natural-language search in any of the 100+ languages the model supports
- ✅ Timeline of moments, preview, manual start/end adjustment
- ✅ 9:16 MP4 export (1080x1920, H.264/AAC) with standardized file names

## Phase 2 — Smarter curation

- ✅ Editorial score (similarity, audio intensity, motion, reaction, novelty, duration fit)
- ✅ Audio peak and motion detection
- ✅ Zero-shot tags from a configurable concept list (`configs/tags.yaml`)
- ✅ Novelty: moments similar to clips you already made are ranked lower
- ⬜ Local vision-language model (e.g. SmolVLM, Qwen2.5-VL) for real scene descriptions
- ⬜ Smarter boundaries: snap clip edges to scene cuts and to quiet moments in the audio
- ⬜ Highlight discovery without a query ("show me the 10 most intense moments")
- ⬜ Learn from your approvals and rejections to tune the score weights for your channel
- ⬜ Webcam-region detection so reaction presence uses the face camera directly

## Phase 3 — Social production

- ✅ Approval queue with full history
- ✅ Optional burned-in subtitles (local faster-whisper)
- ✅ Automatic thumbnail
- ⬜ Caption/hashtag suggestions written by a local model, always editable
- ⬜ Cover templates and safe-zone overlays for TikTok, Reels and Shorts
- ⬜ Multiple export presets (Shorts 60 s, Reels 90 s, square, 4:5)
- ⬜ Split-screen layout: gameplay on top, face camera below
- ⬜ Assisted manual publishing: open the platform's own upload page with the file and caption
  ready. Uploading stays a human action.

## Phase 4 — Scale

- ⬜ Batch indexing of a folder of recordings
- ⬜ Clip library across videos with tags and search
- ⬜ Search by reference image or reference audio (the model and backend already support it;
  the UI does not yet)
- ⬜ Multiple channels/profiles with their own presets
- ⬜ Local reports: approval rate per query, time saved per video
- ⬜ GPU-friendly packaging (Docker image with CUDA, one-click Windows installer)

## Explicitly out of scope

These will not be added, by design (see [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md)):

- Downloading videos from links or platforms
- Automatic publishing without a human decision
- Cloud processing or any upload of footage, audio, frames or embeddings
- Telemetry
- Claims or "scores" about virality
