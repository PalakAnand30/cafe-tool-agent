.DEFAULT_GOAL := help

UV ?= uv
export UV_CACHE_DIR ?= $(CURDIR)/.cache/uv

.PHONY: help setup lint test data push-data eval-local report serve docker-build docker-run

help:
	@echo "Available now: setup, lint, test"
	@echo "Later phases: data, push-data, eval-local, report, serve, docker-build, docker-run"

setup:
	$(UV) sync --locked

lint:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .

test:
	$(UV) run --locked pytest

data push-data:
	@echo "$@ is not implemented yet; scheduled for Phase 2." >&2
	@exit 2

eval-local:
	@echo "$@ is not implemented yet; harness in Phase 3, GGUF evaluation in Phase 5." >&2
	@exit 2

report:
	@echo "$@ is not implemented yet; scheduled for Phase 5." >&2
	@exit 2

serve docker-build docker-run:
	@echo "$@ is not implemented yet; scheduled for Phase 6." >&2
	@exit 2
