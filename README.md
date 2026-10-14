# Café Ordering Agent

A reproducible project to fine-tune Qwen2.5-1.5B-Instruct with QLoRA for café
ordering tool calls, compare prompting and fine-tuning, and serve the resulting
GGUF model on CPU. `PROJECT_SPEC.md` defines the full scope and working rules.

Only the Phase 0 scaffold is available. Training, data generation, evaluation
and serving have not run. No model performance results are available.

## Local setup

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
make setup
make lint
make test
```

The src-layout distribution is named `cafe-tool-agent`; its Python import is
`cafe_agent`. The Python version and direct dependency versions are pinned;
`uv.lock` records the resolved environment. Local setup installs no GPU libraries.

When a later phase needs credentials, copy `.env.example` to `.env` only if
you do not already have one, then fill in the required values locally.
Never commit or share `.env`.

## Project layout

- `src/cafe_agent/`: empty modules for domain, backend, tools, prompting,
  parsing, agent, model clients, data, evaluation and serving.
- `configs/`: YAML placeholders, with seed 42 reserved for reproducible jobs.
- `tests/`: installed-package smoke test.
- `notebooks/`, `scripts/`, `docker/`: placeholders for their planned phases.
- `data/`: generated data and API cache are ignored; human tests and small
  samples can be versioned.
- `results/`: versioned location for future predictions and reports.
- `.github/workflows/ci.yml`: locked setup, lint and tests on pushes and PRs.
- `AGENTS.md` and `PROGRESS.md`: working rules, decisions and phase status.

## Commands and phase boundaries

`make setup`, `make lint` and `make test` are implemented. Remaining targets
report the phase that will implement them and exit with an error:

| Target | Planned phase |
| --- | --- |
| `make data`, `make push-data` | 2 |
| `make eval-local` | 3 (harness), 5 (GGUF) |
| `make report` | 5 |
| `make serve`, `make docker-build`, `make docker-run` | 6 |

GPU training and inference run only on Kaggle T4 notebooks. Local GGUF
inference will use CPU only (`n_gpu_layers=0`).

<!-- RESULTS:START -->
No experiments have run; results will be generated from `results/` in later phases.
<!-- RESULTS:END -->

Continue one phase at a time, following Section 0 of `PROJECT_SPEC.md`.
