.DEFAULT_GOAL := help

UV ?= uv
export UV_CACHE_DIR ?= $(CURDIR)/.cache/uv
export TOKENIZERS_PARALLELISM ?= false
export PYTHONHASHSEED ?= 42

.PHONY: help setup setup-cpu lint test benchmark data push-data eval-local eval-s3 latency report serve docker-build docker-run

help:
	@echo "Available: setup, setup-cpu, lint, test, benchmark, data, push-data, eval-local, eval-s3, latency, report"
	@echo "Phase 6: serve, docker-build, docker-run"

setup:
	$(UV) sync --locked

setup-cpu:
	$(UV) run --locked python scripts/setup_cpu.py

lint:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .

test:
	$(UV) run --locked pytest

data:
	$(UV) run --locked python -m cafe_agent.data.generator

benchmark:
	$(UV) run --locked python scripts/convert_independent_test.py

push-data:
	$(UV) run --locked python scripts/push_dataset.py

eval-local:
	$(UV) run --locked --extra cpu python -m cafe_agent.cpu.evaluate

eval-s3:
	$(UV) run --locked python scripts/evaluate.py

latency:
	$(UV) run --locked --extra cpu python -m cafe_agent.eval.latency

report:
	$(UV) run --locked python scripts/make_report.py

serve docker-build docker-run:
	@echo "$@ is not implemented yet; scheduled for Phase 6." >&2
	@exit 2
