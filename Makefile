.PHONY: install dev test test-ollama lint format typecheck dataset train evaluate check up down

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
	uv run mypy app tests

dataset:
	uv run python -m app.ml.training.dataset

train: dataset
	uv run python -m app.ml.training.train

evaluate:
	uv run python -m app.ml.training.evaluate --dataset data/processed/routing_synthetic_v1.csv

check: lint typecheck test

up:
	docker compose up --build

down:
	docker compose down
