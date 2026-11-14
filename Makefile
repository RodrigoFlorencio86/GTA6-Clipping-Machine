# Convenience targets (Linux/macOS, or Windows with make + Git Bash).
# On Windows without make, run the commands in each target directly.

PYTHON ?= python
VENV   ?= .venv
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
else
	BIN := $(VENV)/bin
endif

.PHONY: help venv install dev run doctor model test test-model lint format typecheck check docker docker-gpu

help:
	@echo "install     create .venv and install the app"
	@echo "dev         install with development tools"
	@echo "run         start the web app on http://127.0.0.1:8501"
	@echo "doctor      check FFmpeg, PyTorch and the model cache"
	@echo "model       download EmbeddingGemma 2 once (~1.5 GB)"
	@echo "test        run the test suite (no model download)"
	@echo "test-model  run the opt-in test with the real model"
	@echo "check       lint + format check + type check + tests"
	@echo "docker      build and run with Docker Compose (CPU)"

venv:
	$(PYTHON) -m venv $(VENV)

install: venv
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e .

dev: venv
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

run:
	$(BIN)/clipping-machine run

doctor:
	$(BIN)/clipping-machine doctor

model:
	$(BIN)/clipping-machine download-model

test:
	$(BIN)/python -m pytest

test-model:
	$(BIN)/python -m pytest -m model

lint:
	$(BIN)/python -m ruff check src tests

format:
	$(BIN)/python -m ruff format src tests

typecheck:
	$(BIN)/python -m mypy

check: lint typecheck test
	$(BIN)/python -m ruff format --check src tests

docker:
	docker compose up --build app

docker-gpu:
	docker compose --profile gpu up --build app-gpu
