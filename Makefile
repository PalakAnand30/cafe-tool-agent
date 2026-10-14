.DEFAULT_GOAL := help

UV ?= uv
export UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
export TOKENIZERS_PARALLELISM ?= false

.PHONY: help setup lint test data push-data eval-local report serve docker-build docker-run

help:
	@echo "Available now: setup, lint, test, data, push-data"
	@echo "Later phases: eval-local, report, serve, docker-build, docker-run"

setup:
	$(UV) sync --locked

lint:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .

test:
	$(UV) run --locked pytest

data:
	$(UV) run --locked python -m cafe_agent.data.generator

push-data:
	$(UV) run --locked python scripts/push_dataset.py

eval-local:
	@echo "$@ is not implemented yet; harness in Phase 3, GGUF evaluation in Phase 5." >&2
	@exit 2

report:
	@echo "$@ is not implemented yet; scheduled for Phase 5." >&2
	@exit 2

serve docker-build docker-run:
	@echo "$@ is not implemented yet; scheduled for Phase 6." >&2
	@exit 2
