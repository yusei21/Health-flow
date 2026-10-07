.PHONY: install dev test test-ollama lint format typecheck check up down \
	dataset train evaluate dataset-synthetic train-synthetic evaluate-synthetic \
	dataset-mimic train-mimic evaluate-mimic benchmark-ml benchmark-summary

install:
	uv sync

dev:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest

test-ollama:
	HEALTHFLOW_RUN_OLLAMA_TESTS=1 uv run pytest -m ollama

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy app tests benchmarks/scripts

check: lint typecheck test

# --- ML -------------------------------------------------------------------
# Paths are overridable, e.g. `make dataset-mimic MIMIC_TRIAGE=/path/triage.csv`.
MIMIC_TRIAGE ?= data/raw/mimic-iv-ed
MIMIC_SOURCE_VERSION ?= unspecified
MIMIC_DATASET ?= data/processed/routing_mimic_v1.csv
MIMIC_MODEL_DIR ?= models/mimic-structured-v1
BENCHMARK_DIR ?= benchmarks/results
TRAIN = uv run python -m app.ml.training.train --benchmark-dir $(BENCHMARK_DIR)
EVALUATE = uv run python -m app.ml.training.evaluate

dataset-synthetic:
	uv run python -m app.ml.training.dataset

train-synthetic: dataset-synthetic
	$(TRAIN) --experiment synthetic_baseline

evaluate-synthetic:
	$(EVALUATE) --experiment synthetic_baseline

dataset-mimic:
	uv run python -m app.ml.training.prepare_mimic --input $(MIMIC_TRIAGE) \
		--output $(MIMIC_DATASET) --source-version $(MIMIC_SOURCE_VERSION)

train-mimic:
	$(TRAIN) --experiment structured_mimic_baseline --dataset $(MIMIC_DATASET) \
		--model-dir $(MIMIC_MODEL_DIR)

evaluate-mimic:
	$(EVALUATE) --experiment structured_mimic_baseline --dataset $(MIMIC_DATASET) \
		--model-dir $(MIMIC_MODEL_DIR)

# Experiment A always; experiment B only when the processed MIMIC dataset exists.
benchmark-ml: train-synthetic
	@if [ -f $(MIMIC_DATASET) ]; then $(MAKE) train-mimic; \
	else echo "skipping MIMIC: $(MIMIC_DATASET) not found (run make dataset-mimic)"; fi

benchmark-summary:
	uv run python benchmarks/scripts/summarize_results.py $(BENCHMARK_DIR)

# Backwards-compatible aliases (synthetic baseline served by the API).
dataset: dataset-synthetic
train: train-synthetic
evaluate: evaluate-synthetic

up:
	docker compose up --build

down:
	docker compose down
