# Project progress

Updated: 2026-10-14 (Europe/London).

## Phase checklist

- [x] Phase 0: scaffold.
- [x] Phase 1: domain, backend, tools, parser and agent loop.
- [ ] Phase 2: dataset and human-test annotation workflow.
- [ ] Phase 3: evaluation harness and Groq baseline.
- [ ] Phase 4: Kaggle training and GPU evaluation notebooks.
- [ ] Phase 5: GGUF evaluation and full report.
- [ ] Phase 6: demo and deployment.
- [ ] Phase 7: polish and reproducibility.

## Phase 0 scope and decisions

- Read `PROJECT_SPEC.md` fully before implementation. No earlier repository or
  applicable parent `AGENTS.md` was present. Preserve the supplied spec unchanged.
- Scaffold the src-layout `cafe_agent` package, configurations, data/results
  directories, notebooks/scripts/docker placeholders, Makefile and CI.
- Target Python 3.11 (`.python-version`), using the existing uv-managed interpreter.
- Install only foundational local dependencies and development tools in this
  phase. Backend, API, tokenizer, inference, reporting and UI dependencies will
  be pinned when their phases need them. No GPU dependencies locally.
- Use the standard library `sqlite3` backend in Phase 1.
- Keep later-phase Makefile targets explicit failures until implemented.
- Keep uv's cache in `.cache/uv` under the project; ignore it in Git.
- Resolve dependencies with a release cutoff of 2026-09-30 UTC, at least two
  weeks before the phase date. `uv.lock` pins transitive dependencies.
- Initialise a local Git repository on `main`; do not add a remote or push.

## Dependency versions

Conservative, established releases; release dates verified on PyPI. Local
foundation imports, lint, tests and distribution builds pass with Python
3.11.15 and uv 0.11.26. Future notebook and model stacks have not been validated.

| Dependency | Pin | Release date | Evidence |
| --- | --- | --- | --- |
| Pydantic | 2.11.7 | 2025-06-14 | [PyPI](https://pypi.org/project/pydantic/2.11.7/) |
| python-dotenv | 1.1.1 | 2025-06-24 | [PyPI](https://pypi.org/project/python-dotenv/1.1.1/) |
| PyYAML | 6.0.2 | 2024-08-06 | [PyPI](https://pypi.org/project/PyYAML/6.0.2/) |
| pytest | 8.4.1 | 2025-06-18 | [PyPI](https://pypi.org/project/pytest/8.4.1/) |
| Ruff | 0.12.2 | 2025-07-03 | [PyPI](https://pypi.org/project/ruff/0.12.2/) |
| Hatchling (build backend) | 1.27.0 | 2024-12-15 | [PyPI](https://pypi.org/project/hatchling/1.27.0/) |

## Verification

- `uv lock`: resolved the local stack with the release cutoff; archive upload
  timestamps in the lockfile were checked against that cutoff.
- `make setup`: installed the locked environment with Python 3.11.15. Confirmed
  Pydantic v2 validation, YAML parsing and python-dotenv imports work together.
- `make lint`: passed Ruff lint and format checks (12 Python files).
- `make test`: passed (1 installed-package smoke test, importing all modules
  from a clean subprocess outside the checkout without GPU dependencies).
- `uv build --offline` (using the project cache): built the source distribution
  and a wheel from that source distribution. Checked that all 11 package
  modules are in the wheel and both archives exclude `.env`, virtualenvs,
  local caches and Git metadata.
- Confirmed `.env`, `data/cache/` and `data/generated/` are Git-ignored;
  curated human-test and sample directories remain trackable. The model
  download ignore rule is root-anchored so `cafe_agent.models` is included.
- `make data`: exits with code 2 and a Phase 2 explanation, as intended.
- `git diff --check`: passed. Stage and commit only the scaffold and supplied
  spec; the local `.env` remains untracked.
- CI is configured for pushes and pull requests, but has not run on GitHub.
- No dataset generation, model inference, training or evaluation has run.

## Phase 1 implementation and decisions

- Re-read `AGENTS.md`, this file and the full spec before implementation;
  followed Sections 0, 4 and the Phase 1 scope in Section 11.
- Added a validated menu with 25 items across all five categories, three fixed
  out-of-stock items, near-duplicate latte names, per-size drink prices,
  unsized food prices and per-item modifier deltas.
- Added Pydantic domain/order/result models and a transactional SQLite backend.
  Monetary arithmetic uses Decimal; JSON money fields are decimal strings.
- Order progress uses persisted logical minutes, advanced explicitly with
  `advance_time`. Read operations never advance time. Per-order ETA jitter is
  deterministically seeded from the configured seed and order ID.
- Creation is atomic across all lines. Cancellation is allowed before ready;
  repeated cancellation is idempotent. Snapshots include every order and the clock.
- Implemented five shared function schemas with inlined definitions, one
  argument registry, in-process/HTTP executors and FastAPI REST/tool routes.
  Errors include all six spec codes plus invalid arguments, invalid size,
  ambiguous item, unknown tool and exhausted five-digit order-ID capacity.
- Added repair-free Qwen-tag, bare-JSON and native-call parsing, including
  multiple calls, mixed text, duplicate-key rejection, schema validation,
  truncated-output detection and protocol ID checks.
- Added shared behaviour/menu prompt and a tokenizer-only rendering interface.
  The caller supplies the tokenizer; downloading and measuring it remains Phase 2.
- Added a bounded agent loop with complete-response validation before execution,
  matching call/result IDs, caller-owned history and an execution trace.
- Implemented Groq native calls with atomic on-disk response caching by request
  hash and bounded 429/5xx exponential retries with seeded jitter. Credentials
  load via python-dotenv only on cache miss; cached runs need no credentials.
- Implemented a lazy `LlamaCppClient` fixed to `n_gpu_layers=0`, using the shared
  Qwen renderer. Added explicit HF/vLLM notebook stubs without local GPU imports.
- Added `configs/runtime.yaml` for decoding, retry, cache, CPU, API and round
  limits. Bundled all YAML configs in the wheel for installs outside this checkout.

## Phase 1 dependency pins

All release dates were verified on PyPI and precede the release cutoff.
Installed dependencies were tested together; llama.cpp remains an optional,
uninstalled extra, with its client tested using an injected fake engine.

| Dependency | Pin | Release date | Evidence |
| --- | --- | --- | --- |
| AnyIO | 4.9.0 | 2025-03-17 | [PyPI](https://pypi.org/project/anyio/4.9.0/) |
| FastAPI | 0.115.12 | 2025-03-23 | [PyPI](https://pypi.org/project/fastapi/0.115.12/) |
| Groq SDK | 0.28.0 | 2025-06-12 | [PyPI](https://pypi.org/project/groq/0.28.0/) |
| HTTPX | 0.28.1 | 2024-12-06 | [PyPI](https://pypi.org/project/httpx/0.28.1/) |
| Uvicorn | 0.34.3 | 2025-06-01 | [PyPI](https://pypi.org/project/uvicorn/0.34.3/) |
| llama-cpp-python (optional `cpu`) | 0.3.9 | 2025-05-08 | [PyPI](https://pypi.org/project/llama-cpp-python/0.3.9/) |

AnyIO is pinned explicitly because the newest allowed transitive release emitted
a deprecated alias warning from the chosen Starlette release under our strict
warnings-as-errors test policy. The compatible pin passes without warning filters.

## Phase 1 verification

- `make setup`: passed with the locked Python 3.11 local stack; no GPU packages
  or optional llama.cpp engine installed. Lockfile release timestamps checked.
- `make test`: 123 passed, including every error code, atomic failures, exact
  totals, persistence, deterministic replay, concurrency, cancellation at state
  boundaries, parser adversarial cases, HTTP parity and agent feedback/limits.
- Groq tests use the real SDK with HTTPX MockTransport to verify native payloads,
  cache-key inputs, cache hits, credential-free replay, transient retries and
  permanent-error handling. No live Groq request was made.
- CPU-client tests verify lazy construction, `n_gpu_layers=0`, deterministic
  decoding and the shared renderer with a fake engine/tokenizer. No real model
  inference, tokenizer download, training or dataset generation was performed.
- `uv build --offline`: built source and wheel distributions. Imported every
  package module from the wheel, loaded its bundled configs and exercised its
  in-memory backend. Archive checks excluded secrets, caches, virtualenvs and Git.
- Final `make lint`: passed lint and format checks (32 Python files).
- Final `make test`: 123 passed with warnings treated as errors.
- `git diff --check`: passed; the supplied spec is unchanged, and secrets,
  cache files and runtime databases remain ignored.

## Known issues and user actions

- No blocking issues. No accounts, uploads or secret values are required for Phase 1.
- The sandbox required permission for PyPI access and writes to `.git`.
- uv warns that its in-project cache may enter source distributions; archive
  inspection confirmed the ignore rules exclude it.
- The existing local `.env` must remain untracked and must not be printed.
- Actual Groq service/model availability, tokenizer lengths and GGUF inference
  remain unverified; those checks belong to their later phases. Prompt size must
  be checked against the full rendered-token budget during Phase 2 validation.

## Next step

Stop after the Phase 1 local commit. Start Phase 2 only after the user says
`continue to Phase 2`. No remote has been added and nothing has been pushed.
