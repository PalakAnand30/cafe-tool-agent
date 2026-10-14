# Project working rules

`PROJECT_SPEC.md` is the source of truth. Read it in full, together with this
file and `PROGRESS.md`, when starting a session. Re-read the relevant spec
section at the start of each phase. Resolve conflicts in favour of the spec.

## Phase boundaries

- Work on one phase at a time (Phases 0–7 in Section 11).
- End every phase by running `make lint` and `make test`, recording actual
  results, decisions, known issues and the next step in `PROGRESS.md`, and
  making a local conventional commit (for example `feat(scaffold): add project skeleton`).
- Then stop. Report what was built, test results, user actions as a numbered
  checklist, and open questions. Wait for `continue to Phase N` before the next phase.
- Do not push to GitHub unless the user requests it.

## Environment and dependencies

- Target Python 3.11 with `uv`; local hardware is macOS Apple Silicon with
  16 GB RAM and no CUDA GPU.
- All training and GPU inference run in user-run Kaggle T4 notebooks.
- Never install or import CUDA torch, bitsandbytes, vllm, unsloth, mlx or mlx-lm
  locally. Do not use MLX anywhere. Keep GPU dependencies inside notebooks only.
- Local work includes backend, data generation, Groq API, tokenizer-only checks,
  metrics, reports, CPU inference, demo, Docker and CI. Configure llama.cpp with
  `n_gpu_layers=0`.
- Pin every direct dependency exactly in `pyproject.toml`, commit `uv.lock`,
  and pin notebook install cells. Choose compatible releases at least two
  weeks old; record versions and evidence in `PROGRESS.md`.
- Add planned-stack dependencies as their phases need them. Ask before
  installing anything outside the planned stack.
- Keep `cafe_agent` installable from GitHub so Kaggle reuses the same prompt,
  parser, tools and evaluation code.

## Commands

- `make setup`: install the locked local environment (including development tools).
- `make lint`: Ruff lint and formatting checks.
- `make test`: pytest, using the installed package without a source-path override.
- `make data`, `make push-data`: Phase 2.
- `make eval-local`: Phase 3 harness, extended for CPU GGUF in Phase 5.
- `make report`: Phase 5.
- `make serve`, `make docker-build`, `make docker-run`: Phase 6.
- Later-phase targets fail explicitly until implemented; do not report them as working.

## Secrets, APIs and resumability

- Read `.env` with python-dotenv. Expected keys: `GROQ_API_KEY`, `HF_TOKEN`,
  `HF_USERNAME`, `WANDB_API_KEY`, `GITHUB_USERNAME`. Never print, log or commit
  their values. Keep `.env` ignored and `.env.example` values empty.
- Kaggle reads `HF_TOKEN` and `WANDB_API_KEY` with
  `kaggle_secrets.UserSecretsClient`; a private GitHub repo also needs its
  optional `GITHUB_TOKEN` Kaggle secret.
- Cache every external Groq response in `data/cache/`, keyed by a hash of
  model, messages, tools and parameters. Re-runs must reuse cache entries.
- Use exponential backoff with jitter for 429/5xx. Long jobs use append-only
  JSONL and skip completed IDs on resume.

## Conventions and evidence

- Use small, focused modules, type hints and Pydantic v2 structured models.
- Use Ruff lint and formatting, and pytest. Put configuration in
  `configs/*.yaml`, with no magic numbers. The global seed is 42 everywhere.
- Use `sqlite3` from the standard library for the backend; do not add SQLModel
  unless a later decision justifies it.
- Never invent results. README numbers, charts and CV bullets must come from
  code reading `results/`. State clearly when an experiment has not run.
