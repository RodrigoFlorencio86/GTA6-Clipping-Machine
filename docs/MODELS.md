# Models

## EmbeddingGemma 2 (required)

[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) is an open
multimodal embedding model from Google DeepMind (Apache-2.0). It maps text, images, video and
audio into **one shared 768-dimensional space**, so a sentence such as "police chase at night"
can be compared directly with an 8-second piece of gameplay.

| | |
|---|---|
| Parameters | 740M total: 270M text backbone, 170M vision encoder, 300M audio encoder |
| Output | 768 dimensions; Matryoshka truncation to 512 / 256 / 128 |
| Context | 8,192 tokens shared by every modality |
| Languages | 100+ (queries in Portuguese, Spanish, etc. work without translation) |
| License | Apache-2.0 |
| Download | ~1.5 GB, cached under `~/.cache/huggingface/` |

This project uses it **only** to compute embeddings for retrieval. It does not generate text,
transcribe speech or make editing decisions.

### Configuration

`configs/models.yaml`:

```yaml
embedding:
  model: google/embeddinggemma-2   # Hub id, or a local folder with the weights
  dimension: 768                   # or 512 / 256 / 128 (re-index after changing)
  normalize: true
  device: auto                     # auto | cuda | mps | cpu
  dtype: auto                      # bfloat16 on CUDA GPUs that support it, else float32
  batch_size: 4
  include_audio: true              # false = skip the audio encoder entirely
  local_files_only: false          # true = never contact the Hub
  query_prompt_name: SearchQuery
  vision_tokens: auto              # 70 on CPU, 140 on GPU; 280+ for finer detail
```

Environment overrides: `GCM_EMBEDDING_MODEL` (e.g. a folder you copied the weights to),
`GCM_DEVICE`, `GCM_OFFLINE=1` (sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` and
`local_files_only`).

First download, then go offline:

```bash
clipping-machine download-model     # downloads and runs one test embedding
export GCM_OFFLINE=1                # PowerShell: $env:GCM_OFFLINE = "1"
```

### How the app feeds the model

Verified with sentence-transformers 6.3 and transformers 5.23.

- **Queries** are plain text encoded with `prompt_name="SearchQuery"` (`task: search result | query: `).
- **Windows**: one input per 8-second window, built as
  `{"video": [4 PIL frames], "audio": {"array": float32[128000], "sampling_rate": 16000}}`.
  - Frames are the keyframes FFmpeg already extracted (one every 2 s), passed as decoded RGB
    images of the same size, with `do_sample_frames=False`. No torch video decoder is needed,
    and the frames are not re-sampled.
  - Audio is mono, 16 kHz, 1-D float32. The library does **not** resample or validate the rate,
    so the client refuses anything else instead of producing silently wrong embeddings.
    Silent or missing audio is left out and the window is embedded from its frames.
  - Media inputs get no task prompt.
- **Precision**: never float16 (the model card documents NaNs). Non-finite outputs are rejected.
- **Matryoshka**: with `dimension < 768` the client asks for `truncate_dim` and re-normalizes.
  Query and index must use the same dimension; the index keeps one table per dimension.

### Measured behaviour

Measured on the author's laptop (AMD Ryzen 7 5825U, 8 cores, no CUDA GPU, 40 GB RAM, CPU
float32). Your numbers will differ.

**Similarity range.** Cosine similarity between a text query and a video window sits in a narrow
band:

| Relation | Typical cosine |
|---|---|
| Unrelated content | 0.50 – 0.62 |
| Clear match | 0.68 – 0.78 |

That is why `min_similarity: 0.65` is a sensible default for the raw cosine, and why the
editorial score rescales the band [0.55, 0.80] to [0, 1] (`search.calibration`): otherwise the
semantic term of the score would barely vary. In a test with 8 distinct scenes, the correct
scene was the top match for 8 of 8 English queries and for the Portuguese ones ("um gato",
"xícara de café", "foguete na plataforma de lançamento").

**Audio** changes a window's embedding only slightly (cosine 0.95–0.98 between the same frames
with and without audio). It helps queries about sound, but frames dominate.

**Speed (CPU)**, per 8-second window with 4 frames:

| Vision tokens per frame | Seconds per window | Top-1 accuracy (8-scene test) |
|---|---|---|
| 140 | ~12 | 8/8 |
| 70 (CPU default) | ~6 | 8/8 |
| 2 frames × 70 | ~3.7 | 8/8 |

End to end, a 96-second video indexed in **113 s on CPU (≈1.2× its duration)**, including scene
detection, keyframes and audio. One hour of gameplay therefore takes roughly 1–1.5 hours on a
similar CPU. A CUDA GPU is much faster; with `device: auto` it is used automatically.

Search is fast once the model is loaded: about **0.2 s** per query on CPU. Loading the model
takes 15–50 s and happens once per app start (or on the first search/indexing).

**Memory.** The model in float32 needs roughly 3–4 GB of RAM. 16 GB of system RAM is enough;
avoid running several indexing processes at once.

### Tuning tips

- Slow machine: keep `vision_tokens: 70`, or set `include_audio: false` (frames only).
- Fine detail matters (HUD text, small objects): `vision_tokens: 280` on a GPU.
- Huge libraries: `dimension: 256` cuts index size by 3× with a small quality loss.
- Change `configs/tags.yaml` to describe what *your* gameplay looks like. Tags are just
  embedded sentences, so you can add your own concepts in any language and re-index.

## faster-whisper (optional)

Only used when you tick "Burn in subtitles" on export. It transcribes the clip's audio
locally. Install with `pip install -e ".[subtitles]"`. Configure the model size and language in
`configs/models.yaml` → `subtitles`. Its weights are downloaded on first use (block that with
`GCM_OFFLINE=1` once they are cached).

## Not used

- No cloud APIs and no hosted models.
- No vision-language model yet. Real scene descriptions with a local VLM are on the
  [roadmap](ROADMAP.md).
