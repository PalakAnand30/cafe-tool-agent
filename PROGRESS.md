# Project progress

Updated: 2026-10-14 (Europe/London).

## Phase checklist

- [x] Phase 0: scaffold.
- [x] Phase 1: domain, backend, tools, parser and agent loop.
- [x] Phase 2: dataset and human-test annotation workflow.
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

## Phase 2 implementation and approved adjustments

- Re-read the rules and full specification; implement Phase 2 only.
- User approved raising the full-conversation limit from 1,024 to 4,096 tokens
  while retaining full domain coverage. The pinned tokenizer measured 2,555
  tokens for the original prompt before a reply, and 1,023 even after an initial
  full-menu/schema compaction. Updated Sections 5, 7 and 7b of the source spec,
  training max length and inference context (8,192) accordingly. T4 memory
  sizing remains a Phase 4 check.
- User approved replacing unavailable `llama-3.3-70b-versatile` with
  `openai/gpt-oss-120b` through Groq for paraphrasing and the future S3 baseline.
  The original call returned `model_not_found`; its ID was absent from the key's
  cached model catalog. The replacement's live connectivity/paraphrase check
  succeeded. Updated active config, spec and dataset provenance; preserve
  historical Phase 1 test descriptions as history.
- Use low reasoning effort, temperature 0 and seed 42. The replacement is
  documented in [Groq's model page](https://console.groq.com/docs/model/openai/gpt-oss-120b)
  and [official OpenAI documentation](https://developers.openai.com/api/docs/models/gpt-oss-120b).
  Applied the OpenAI Docs skill for the explicit model replacement. Requests
  still use the Groq SDK/account, not the OpenAI API.
- Pinned Qwen model/tokenizer revision:
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Only tokenizer/config files were
  downloaded into the ignored project cache; no model weights or torch.
- Compact the shared prompt by grouping identical per-item modifier sets;
  retain every item/modifier, and obtain prices/stock from tools. Remove only
  schema titles/default metadata, retaining actual validation constraints.
- Added the Pydantic conversation contract, per-assistant gold behaviours,
  entity provenance, scripted user turns, logical-clock advances and author tags.
- Implemented five wording families per intent, programmatic backend-derived
  labels/results/state and varied final replies. Generation targets the spec's
  split sizes, intent shares and multi-turn rate, with deterministic per-ID seeds.
- Reserve one of five template families per intent (20%) and configured
  item/modifier combinations exclusively for test. Deduplicate normalized user
  text across all synthetic conversations, before and after text transforms.
- Batch Groq paraphrases, preserving entities and complete order-line phrases
  to prevent swapped quantity/modifier bindings. Reject missing/new entities,
  changed bindings, new negations, malformed outputs and unchanged rewrites.
  Record failures and fall back to original text. Native responses remain
  content-addressed and atomically cached by the shared Groq client.
- Surface noise preserves order facts. Drafts, paraphrase journals and final
  split files are append-only and skip completed IDs. Fingerprinted manifests
  reject changed source/config inputs. Interrupted JSONL tails are backed up
  before repair; malformed complete records fail explicitly.
- Every record checks shared schemas/prompt, call/result protocol, exact real
  backend replay and final state, and full Qwen token length. Assistant-only
  labels use template-prefix spans and token offsets, including closing tokens;
  no truncation, JSON repair or framework/tensor imports.
- Added the interactive human CLI with explicit gold actions and expected-state
  review/input, fresh backends, optional time advances, replay validation and
  append-only saves. Seeded five `codex_example` records, excluded by the shared
  scored-human loader. No human-authored conversations were invented.
- Added private Hub publication and a local dry-run packet with default synthetic
  and separate human-test configs, dataset card, manifest and generation stats.
  Upload checks human benchmark size and rejects existing public repositories.
  Raw JSONL is canonical to preserve omitted argument keys across HF loading.
- Cache parsed YAML by file contents and return copies, avoiding repeated parsing
  during replay while reflecting edits and preventing caller mutation of cache.
- Added the pinned tokenizer cache/download to CI and implemented `make data`
  and user-run `make push-data`.

## Phase 2 dependency pins

| Dependency | Pin | Release date | Evidence |
| --- | --- | --- | --- |
| Transformers (tokenizer only locally) | 4.51.3 | 2025-04-14 | [PyPI](https://pypi.org/project/transformers/4.51.3/) |
| Hugging Face Hub | 0.30.2 | 2025-04-08 | [PyPI](https://pypi.org/project/huggingface-hub/0.30.2/) |
| Jinja2 (chat-template renderer) | 3.1.6 | 2025-03-05 | [PyPI](https://pypi.org/project/Jinja2/3.1.6/) |

All direct pins predate the release cutoff; transitive versions remain in
`uv.lock`. No dependency outside the planned stack was added.

## Phase 2 verification

- `make lint`: passed; final checks include 54 Python files.
- `make test`: 204 passed with warnings treated as errors, including all prior
  backend/parser/client cases and real-tokenizer tests for every intent/family,
  replay corruption, leakage, exact plan shares, masking (including fake role
  delimiters in user text and consecutive tool results), interrupted-tail
  recovery, append-only resume, entity/binding verification, annotation saves,
  worked-example exclusion and private upload preparation.
- `uv build --offline`: source and wheel build passed. Every package module
  imports from the wheel; packaged configs load, GPU imports are absent, and
  source/wheel archive checks exclude secrets, caches, virtualenvs and Git.
- Full `make data`: passed. All 4,200 final conversations validated (100%):
  train 3,500, validation 300, synthetic test 400; exactly 1,050 multi-turn
  conversations and 5,250 customer turns. All tool results and final states replay.
- Measured full-conversation lengths: minimum 1,175, maximum 2,816, mean
  1,325.44 tokens, all below the user-approved 4,096-token limit.
- Paraphrasing attempted on 1,563 turns: 178 verifier rejections (11.39%):
  147 unchanged, 20 missing protected request text, 11 missing order-line bindings.
  One further deduplication fallback leaves 1,384 retained paraphrase turns.
  All rejected candidates use their original utterance. Noise applied to 801 turns.
- Final leakage checks: zero overlapping normalized customer texts; all ten
  held-out template families and all three reserved item/modifier pairs observed
  only in synthetic test.
- Re-ran `make data` successfully inside the network-restricted sandbox: all
  completed IDs/batches were reused, with no new Groq requests.
- `scripts/push_dataset.py --dry-run`: passed after replaying all records;
  staged 4,200 synthetic conversations and five excluded examples. Human scored
  count is zero. No Hub repository was created or changed.
- `scripts/collect_data_artifacts.py` regenerates 20 review records (one per
  intent from train and test), copied statistics and `results/data_validation.json`
  directly from the validated files. Full generated data and cache stay ignored.
- Confirmed the live environment has no GPU packages or optional llama.cpp;
  downloaded HF assets contain no model weights. Lockfile upload timestamps
  obey the release cutoff. Current-source manifest check passes.
- Final `make lint`: passed (54 Python files); final `make test`: 204 passed
  in 4.50 seconds, with warnings treated as errors.
- The dry-run packet matches validated splits, manifest and statistics
  byte-for-byte. Final source/wheel builds pass and exclude the full generated
  corpus and all secrets/caches. `git diff --check` passes.

## Known issues and user actions

- Write the configured 60–100 human-authored conversations with the annotation
  CLI before seeing any model evaluation results, then run `make push-data` to
  publish privately. HF_TOKEN and HF_USERNAME are already present in `.env`;
  their values were not displayed. Dataset-write permission is checked when the
  user actually uploads. These are user actions from Phase 2.
- The sandbox required permission for PyPI access and writes to `.git`.
- uv warns that its in-project cache may enter source distributions; archive
  inspection confirmed the ignore rules exclude it.
- The existing local `.env` must remain untracked and must not be printed.
- The Groq replacement and tokenizer renderer have now been exercised locally.
  No S3 benchmark, GPU training/evaluation or real GGUF inference has run.
- The account's 8,000-token/minute Groq limit slows generation. Parallel cache
  preparation hit that limit and stopped; the canonical resumable run completed
  with its existing bounded exponential backoff. Preserve cache/journals on retry.
- The interrupted pre-review drafts were preserved in the ignored
  `.cache/phase2-interrupted-drafts` directory before the finalized run.

## Next step

Stop after the Phase 2 local commit. Start Phase 3 only after the user says
`continue to Phase 3`; the human benchmark must be written before model
evaluation. No GitHub remote/push or HF upload has been performed.
