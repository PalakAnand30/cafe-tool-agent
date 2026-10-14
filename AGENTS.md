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
- Standalone café API: `uv run --locked uvicorn cafe_agent.backend.app:create_app --factory`.
  This is the backend only; the demo `make serve` target remains Phase 6.
- `make data`: generate and replay/token-validate resumable synthetic splits.
- `make push-data`: user-run private Hub publication after the human benchmark.
- Human annotation: `uv run --locked python scripts/annotate_human_test.py`.
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

## Phase 1 implementation conventions

- Use `CafeService.call` for validated tool dispatch and `snapshot()` for
  reproducible state. New services default to isolated in-memory SQLite.
- State progresses only through explicit `advance_time(minutes)`; status
  observations never mutate the logical clock. Persist the clock with orders.
- Money uses Decimal and serialises to decimal strings; preserve this across
  data generation, replay and both execution transports.
- `tools.schemas` is the single argument/schema registry. The strict parser
  preserves omitted optional arguments and never repairs invalid JSON.
- Use the shared prompt and tokenizer renderer for all systems. Full tokenizer
  and sequence-length checks belong to Phase 2.
- The `cpu` extra is pinned but not installed by normal setup. Keep llama.cpp
  lazy and CPU-only; actual model inference remains unverified until Phase 5.
- Configs are bundled in wheels; checkout configs take precedence locally.

## Phase 2 decisions and data rules

- The user explicitly approved a 4,096-token full-conversation limit, preserving
  full domain coverage. This supersedes the original 1,024-token requirement.
  Inference context is 8,192; measure T4 training memory during Phase 4.
- The user explicitly approved `openai/gpt-oss-120b` through Groq for both
  paraphrasing and S3 after the original Llama model was unavailable. Use low
  reasoning effort. Do not present the replacement as a Llama result.
- Model/tokenizer revision is pinned in `configs/train.yaml`; download only
  tokenizer/config files locally, with no model weights or torch.
- `Conversation` is the shared data contract; gold tool-call arguments are
  dictionaries in JSONL and converted to native wire strings by the renderer.
- Every record must schema-validate, replay to identical tool results/final
  state, and render under the approved token budget. Mask only assistant payloads
  and their closing tokens; no truncation, substring guessing or JSON repair.
- Preserve dataset manifests, append-only drafts, paraphrase journals and final
  records. Changed inputs/code require a new output directory. Keep API cache.
- Reserve test template families and item/modifier pairs before sampling;
  check normalized user text across every synthetic split.
- Human records must be written by the user before seeing model results.
  Always exclude `author=codex_example`; never invent human-authored examples.
- The user runs `make push-data` after annotating the required human benchmark.
  Prepare/dry-run locally; do not upload automatically in this phase.
