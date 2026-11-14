# Contributing

Thanks for helping! Bug reports, test results on other hardware, better default tags and
translations are all very welcome.

## Ground rules

These are part of the project and pull requests that break them will not be merged
(see [docs/ACCEPTABLE_USE.md](docs/ACCEPTABLE_USE.md)):

- No downloading from URLs or platforms. Local files only.
- No automatic publishing or uploading of any kind.
- No cloud processing, no telemetry, no analytics.
- The embedding model is used for embeddings only.
- No claims about virality or guaranteed reach.

`tests/test_guardrails.py` checks several of these automatically.

## Development setup

```bash
git clone https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine.git
cd GTA6-Clipping-Machine
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

FFmpeg must be on `PATH`. The test suite never downloads the model: it uses a deterministic
fake embedder (`tests/fakes.py`).

```bash
python -m pytest                      # all tests
python -m pytest -m model             # opt-in: real EmbeddingGemma 2
python -m ruff check src tests
python -m ruff format src tests
python -m mypy                        # strict typing for src/
```

## Pull requests

- Keep the code typed and covered by tests. New behaviour needs a test.
- Put user-facing text in `src/clipping_machine/ui/i18n.py`, in English and Portuguese.
- New thresholds or weights go in `configs/*.yaml`, validated in `config.py`.
- The UI and CLI call `ClippingService` (`services.py`); keep FFmpeg, SQLite and LanceDB
  access out of the UI.
- Describe how you tested the change, including OS and GPU/CPU if it touches indexing or export.

## Reporting results

If you run the app on a GPU, macOS or Linux, please open an issue with your hardware, video
length and indexing time. It helps everyone pick good defaults.
