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
- `make benchmark`: convert the supplied Claude-authored set with reviewed gold.
- `make push-data`: private Hub publication after independent benchmark validation.
- Human annotation: `uv run --locked python scripts/annotate_human_test.py`.
- `make eval-local`: resumable S5 CPU GGUF evaluation on both test sets.
- `make eval-s3`: historical cached/resumable Groq baseline; not needed for reporting.
- `make setup-cpu`: install the pinned CPU extra with the forced-Metal CMake override.
- `make latency`: separate CPU benchmark with three warm-ups and 100 fixed requests.
- Offline scoring: `uv run --locked python scripts/evaluate.py --metrics-only`.
- `make report`: regenerate the complete five-system comparison offline from results.
- `make serve`: CPU model, chat API and Gradio demo on loopback port 7860.
- `make docker-build`, `make docker-run`: linux/amd64 CPU image and local demo.
- `uv run --locked python scripts/deploy_space.py`: prepare an allowlisted Space
  bundle without uploading; `--deploy` explicitly creates/updates the private Space.
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
- User-approved follow-up replaces the human-written requirement with the supplied
  independent Claude-authored benchmark. Keep all customer messages and `author=claude`;
  record Codex gold-label annotation provenance. Keep the current domain and revise
  gold labels, without adopting the supplied replacement menu or seeded orders.
- Score the separate `independent_test` split. Never describe LLM-written cases as
  human evidence. Always exclude `author=codex_example`. Optional future human data
  must remain genuinely human-authored and be reported separately.
- The user explicitly requested help with the private upload; complete validation
  and the dry run, then run `make push-data`. No GitHub push is authorized.

## Phase 3 evaluation rules

- Use the shared prediction schema for all systems. Evaluate every gold assistant
  turn plus complete agent conversations on fresh backends; retain scripted user
  messages and explicit logical-time advances. Exclude Codex worked examples.
- Freeze prompts, lexical action rules and argument normalization before the
  baseline. Do not tune the evaluator or prompt from observed test failures.
  Plain-text action rules are approximate and do not judge reply factuality.
- Preserve append-only predictions, API cache and immutable inference manifests.
  Changed inference code/input requires a new output directory; never silently
  mix runs. Score/report complete coverage or mark partial results explicitly.
- Report state-only end success separately from the stronger action-and-state
  success. Empty state alone is insufficient evidence of correct clarification,
  decline or lookup behaviour. Preserve extra-order/cancellation side effects.
- Bootstrap complete independent cases by conversation, not individual turns.
  Keep Claude/Codex provenance and coverage limitations explicit.
- S3 request latency includes network/retries, excludes explicit rate pacing and
  local cache hits. Cost is a dated paid-equivalent estimate, not an actual bill.

## Phase 4 notebook rules and handoff

- `notebooks/train_kaggle.ipynb` and `eval_gpu_kaggle.ipynb` are user-run Kaggle
  workflows. Do not execute their bootstrap/GPU cells on the Mac. GPU packages
  remain notebook-only; package modules under `cafe_agent.kaggle` import them lazily.
- Preserve `configs/train.yaml`, existing data-generation sources and all S3
  inference files/manifests. Kaggle hyperparameters/pins are in `configs/kaggle.yaml`;
  `train_settings()` composes them with the frozen tokenizer identity. Four S2
  examples and their training provenance are frozen in `configs/few_shot.yaml`.
- Bootstrap an isolated Python 3.11 environment on Kaggle. Install the package
  from the exact user-pushed Git commit; require exact dataset/base/merged revisions.
  Never embed tokens in Git URLs, notebook output or model cards.
- Preserve assistant labels with the custom collator and `skip_prepare_dataset`;
  set `label_names=["labels"]`, fp16 and no bf16. Default memory probe is batch 4
  padded to 4,096. OOM requires an explicit new run with batch 1 / accumulation 16,
  retaining full conversations. Record probe and actual training memory separately.
- GPU evaluation reuses the shared runner/schema, fresh backends and clock.
  Separate backend files/processes; only vLLM installation/load failure triggers
  automatic Transformers fallback. Runtime failures stop for resume. Do not roll
  back newer predictions when restoring an older Kaggle input.
- Keep Hub outputs private. Preserve full checkpoint folders and manifests on
  interruption. Validate complete S1/S2/S4 coverage before bundling; retain S3.
- Local checks: `uv run --locked python scripts/validate_notebooks.py` and
  `uv run --locked python -m cafe_agent.kaggle.preflight --output results/notebook_data_preflight.json`.
  Rebuild with `uv run --locked python scripts/build_notebooks.py` after source-cell edits.
- Exact user steps are in `docs/PHASE4_KAGGLE.md`. GPU training, actual T4 memory,
  GPU inference and GGUF conversion remain unverified until the user runs Kaggle.
