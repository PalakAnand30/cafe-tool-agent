# Project progress

Updated: 2026-10-23 (Europe/London).

## Phase checklist

- [x] Phase 0: scaffold.
- [x] Phase 1: domain, backend, tools, parser and agent loop.
- [x] Phase 2: dataset and human-test annotation workflow.
- [x] Phase 3: evaluation harness and Groq baseline.
- [x] Phase 4: Kaggle training and GPU evaluation notebooks (written and locally validated).
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

## Phase 2 follow-up: independent benchmark and private publication

- User requested help completing benchmark preparation and private upload.
  They explicitly approved replacing the human-written requirement with an
  independently LLM-authored benchmark, then chose to keep the current domain
  and revise benchmark gold actions. Updated the spec and working rules.
- Preserved the supplied 84 Claude-authored cases byte-for-byte at
  `data/human_test/human_test_authoring.yaml`. No customer message was rewritten,
  no LLM case was relabelled human, and the supplied replacement domain was not
  adopted. Existing menu, prices, stock and fresh unseeded backends remain authoritative.
- Initial compatibility audit is in `results/human_authoring_audit.json`.
  Codex reviewed gold actions/intents, with overrides and reasons in
  `data/human_test/gold_overrides.yaml`, pinned to source/domain hashes.
  Revised 45 cases and 50 turn expectations for the current menu.
- Added strict Pydantic authoring/override models, deterministic conversion,
  stale-review rejection, `$LAST_ORDER_ID` resolution, actual tool-error outcome
  checks, programmatic replies, backend replay and tokenizer validation.
  `make benchmark` generates `data/human_test/independent_test.jsonl` with
  `author=claude`, `split=independent_test` and `annotator=codex` provenance.
- All 84 cases validate; token lengths range from 1,175 to 2,801. The measured
  intent distribution is in `results/independent_validation.json`: 34 unsupported,
  16 clarification, seven single-order, one multi-order, two availability,
  five status, six cancellation, five menu, two tool-error and six out-of-scope.
  The successful multi-item-order coverage is limited. Dataset docs explicitly
  describe the LLM wording/annotation bias and do not claim human evidence.
- Preserve the original generated run and API cache. Rebuilt and validated all
  4,200 synthetic records under the current schema/source fingerprint in
  `data/generated-independent-v1/`. All are equal to the original records after
  schema loading; no Groq cache files were created or modified during the rebuild.
  Split sizes, paraphrase/noise statistics and token lengths remain unchanged.
- Publisher verifies independent records against reviewed source/overrides and
  checks zero normalized customer-text overlap with every synthetic split.
  Upload packet includes a separate independent_test config, immutable source,
  overrides and validation report. Five legacy codex_example records remain
  separately available and excluded; actual human-authored count remains zero.
- `make benchmark`, `make data` and the upload dry run passed. `make lint` passed
  (57 Python files); `make test` passed all 210 tests in 4.37 seconds, including
  current-domain gold, unchanged customer text, provenance separation, stale
  review, missing prior order, unexpected tool errors and publication integrity.
- `make push-data` completed after allowing the required network access. The
  dataset was uploaded privately with 4,200 synthetic records, 84 independent
  LLM-authored cases and five excluded examples. A read-only Hub check confirmed
  private visibility and matching remote sizes/content hashes for all 12 staged
  files. Receipt: `results/dataset_upload.json`, Hub revision
  `5a626daa659b2455861e6a8d5a891c6dbe14e36d`.
- No new dependencies, model inference, baseline evaluation, training, public
  publication or GitHub push. Credentials remained undisplayed and Git-ignored.
- Offline source/wheel builds passed. Archive checks confirm the independent
  module and current configs are included while secrets, caches, full synthetic
  datasets and model weights are excluded. `git diff --check` passed.

## Phase 3: evaluation harness complete; live baseline deferred by user

- User authorized Phase 3 only. Re-read the complete spec and working rules.
  No Kaggle notebook, GPU work, demo, chart generation or later phase started.
- Implemented a versioned Pydantic prediction schema shared by local and future
  Kaggle clients. `make eval-local` validates both frozen test sets, evaluates
  every gold assistant turn and runs complete agent conversations against fresh
  in-memory backends. Explicit clock advances anchor to scripted user turns.
- Retain exact shared system prompt, tools, temperature 0, max_tokens 512,
  seed 42 and the approved Groq GPT-OSS 120B model with low reasoning effort.
  Canonicalize tool IDs, JSON and schema defaults consistently across modes.
  Raw responses, usage, cache flags, latency, traces and final states are saved.
- Added append-only prediction checkpoints, interrupted-tail recovery, duplicate
  key/provenance checks, an exclusive writer lock, and manifests pinning data,
  runtime, prompt/tools and inference source hashes. Re-runs skip completed
  units and reuse the existing content-addressed Groq cache. Provider-rejected
  tool generations are cached as invalid outputs; outages/quotas stop the job.
- Implemented exact tool action accuracy, normalized argument exact match,
  field-level multiset F1, clarification/decline precision and recall, validity,
  per-intent metrics, confusion matrices, end success, latency and paid-equivalent
  costs. Undefined denominators are null. Plain-text actions use fixed lexical
  rules frozen before the run; they do not judge reply factuality or which
  detail a clarification asks for. No prompt/classifier tuning from test results.
- Final-state success ignores line/modifier ordering, retaining every order,
  ID, price, quantity, name, status and logical time. Report a separate stronger
  action-and-state success metric, because an empty state can match even when
  a decline/clarification/lookup response is wrong. Extra orders and wrong
  cancellations are tested explicitly.
- Implemented deterministic 1,000-resample conversation-cluster percentile
  bootstrap for complete independent-test metrics. Partial runs have no CIs.
  S3 latency includes network/retries, excludes deliberate quota pacing and
  local cache hits. Cost uses Groq's published paid rates checked 2026-10-14
  (USD 0.15/million input and 0.60/million output tokens), with input-cache
  discounts/free-plan billing excluded; it is not an actual bill. Source:
  https://console.groq.com/docs/model/openai/gpt-oss-120b.
- Metrics recompute offline with
  `uv run --locked python scripts/evaluate.py --metrics-only`, including the
  versioned metrics, coverage, baseline table and README results marker block.
- Full live S3 evaluation started with independent_test first. Groq's headers
  confirmed the account's 8,000 TPM limit. The run then reached the daily token
  quota. `results/groq_quota.json` records a real 429/TPD response: limit 200,000,
  used 198,900 and requested 1,131 tokens. Credentials/org identifiers were
  not logged. The user explicitly chose to keep free limits and save progress
  for a later resume, rather than enabling higher limits.
- Saved coverage (`results/eval_coverage.json`): independent_test has 33/143
  gold assistant predictions and 24/84 complete end-to-end conversations.
  Synthetic test remains 0/876 gold assistant predictions and 0/400 end-to-end
  conversations. `results/metrics.json` and `results/S3_baseline.md` are explicitly
  partial; they are not full benchmark results and have no confidence intervals.
- The latest live retry proved completed IDs are reused; failed quota requests
  were not scored as model predictions. SDK-wrapped quota errors are recognized
  separately from connectivity errors. All frozen inference source hashes still
  match the run manifest. Synthetic files and their generation manifest remain
  unchanged and current; no data regeneration or new dependencies were needed.
- Verification: `make lint` passed (67 Python files). Final `make test` passed 232
  tests in 4.60 seconds; 22 evaluator tests cover perfect/partial/invalid fixtures,
  empty-state false success, extra orders, clock/cancellation progression,
  argument binding, precision/recall, bootstrap determinism, cache-key identity,
  provider failures/quotas, append-only resume, stale manifests, duplicate
  predictions, dataset tampering and fully offline metrics recomputation.
- Final offline source/wheel builds and archive checks passed: evaluator modules
  and config are included, with secrets/caches/full synthetic data/model weights
  excluded. `git diff --check` passed.
- Phase 3 acceptance is **not complete**: the full S3 baseline and complete-set
  confidence intervals are pending quota availability and the user's later resume.
  No GitHub push, HF visibility change, new dataset upload or later phase work.

## Phase 3 conditional quota check and resume (2026-10-15)

- User requested a quota check and authorized resuming Phase 3 only if quota
  was available. Sent one genuinely uncached pending gold-turn request with
  retries disabled for this probe; successful response was saved under the
  normal full-request cache key. Groq returned HTTP 200. Prediction checkpoints
  were unchanged by the check itself. Sanitized evidence, without credentials
  or organization identifiers: `results/groq_quota_check.json`.
- Resumed `make eval-local` only after the successful live quota check. Reused
  all completed keys and cached responses. Prompts, decoding, argument rules,
  classifiers and all frozen inference source hashes remain unchanged.
- Completed independent_test: 143/143 gold assistant turns and 84/84 end-to-end
  conversations. A local integrity check replayed every saved execution trace
  and reproduced its recorded backend state; source IDs and prediction keys
  are complete and unique. Evidence: `results/phase3_resume_validation.json`.
  This check validates collection integrity, not model accuracy.
- The independent-set metrics now include the configured 1,000-resample, 95%
  conversation-cluster bootstrap intervals. The synthetic set progressed to
  59/876 gold assistant turns and 24/400 end-to-end conversations. Its results
  remain explicitly partial and have no confidence intervals.
- Replayed all 108 saved end-to-end histories (84 independent and 24 synthetic),
  including recorded logical-clock advances. Every tool result and final state
  matched. The original checkpoint prefix remains intact, frozen inference
  hashes match, and offline metric recomputation reproduces the same results.
- `make lint` passed (67 Python files); `make test` passed all 232 tests in
  8.58 seconds. No source, dependency, prompt or decoding changes were needed.
- Groq then returned another quota limit with a 119-second retry-after value.
  The evaluator stopped and preserved its checkpoint. No further resume was
  attempted once quota was unavailable, following the user's condition.
  Current stop metadata: `results/eval_status.json`; current coverage:
  `results/eval_coverage.json`. Prior October 14 quota evidence is historical.
- Refreshed metrics, baseline report and README table directly from saved
  prediction files. Full Phase 3 acceptance is still incomplete because the
  synthetic baseline remains unfinished. Keep free limits and resume from
  the same files when quota is available again; do not start Phase 4.

## Phase 3 evening quota check and resume (2026-10-15)

- User requested another quota check, retaining the earlier conditional resume
  instruction and free-plan preference. At 19:46:59 UTC, one uncached pending
  request succeeded with HTTP 200; cached the response without changing prediction
  checkpoints. Latest sanitized evidence: `results/groq_quota_check.json`.
- Resumed `make eval-local` after that check. Synthetic coverage advanced from
  59 to 143 of 876 gold assistant turns and from 24 to 62 of 400 end-to-end
  conversations. Independent coverage remains complete at 143 turns and 84
  conversations; its prediction file is byte-for-byte unchanged.
- Groq returned another quota limit with a 325-second retry-after value. The
  evaluator stopped, saved partial metrics and retained every completed prediction
  and cached response. Did not resume again after this quota stop. Full Phase 3
  acceptance remains incomplete; no later phase started.
- Offline checks replayed all 146 saved end-to-end histories (84 independent,
  62 synthetic), reproducing tool results and recorded backend states. Both
  original checkpoint prefixes are preserved, frozen inference hashes match,
  and offline metrics recomputation is identical. Evidence is recorded in
  `results/phase3_resume_validation.json`.
- `make lint` passed (67 Python files); `make test` passed all 232 tests in
  4.31 seconds. Refreshed reports and README from saved predictions; no source,
  dependency, domain, prompt or decoding changes were made.

## Phase 3 second credential and resume (2026-10-16)

- User supplied `GROQ_API_KEY2` in the ignored `.env`. Tested it with one uncached
  pending evaluation request; Groq returned HTTP 200. Selected the second key
  for this run by loading it with python-dotenv and assigning it to the process's
  `GROQ_API_KEY` before evaluation. Did not edit `.env`, print either credential,
  change default credential selection, or implement automatic key rotation.
- Continued the earlier conditional Phase 3 authorization. Preserved all cached
  responses and completed predictions. A connection error stopped the first
  attempt at 196 synthetic end-to-end conversations. A fresh pending-request
  probe succeeded, so resumed again. Latest sanitized check, including only the
  credential variable name, is in `results/groq_quota_check.json`.
- Synthetic coverage advanced from 143 to 580 of 876 gold assistant turns and
  from 62 to 248 of 400 end-to-end conversations (186 additional conversations).
  Independent coverage remains complete at 143 turns and 84 conversations; its
  predictions are byte-for-byte unchanged.
- Groq then returned a quota limit with a 211-second retry-after value. Stopped
  at that checkpoint without another resume, preserving the free-plan preference.
  `results/eval_status.json` records the stop. Reports and README were regenerated
  from saved predictions and continue to label synthetic results as partial.
- Offline integrity checks replayed all 332 saved end-to-end histories (84
  independent, 248 synthetic), reproducing every tool result and recorded backend
  state. Both previous checkpoint prefixes remain intact, frozen inference hashes
  match, and offline metric recomputation is identical. Evidence:
  `results/phase3_resume_validation.json`. This is collector validation, not an
  accuracy claim. No prompt, domain, model parameter, inference source or
  dependency changes were made.
- `make lint` passed (67 Python files); `make test` passed all 232 tests in
  4.16 seconds. Phase 3 remains incomplete; no later phase started.

## Phase 3 original-key resume and completion (2026-10-16)

- User requested checking the original `GROQ_API_KEY`. A genuinely uncached pending
  request succeeded with HTTP 200 at 17:28:11 UTC. Explicitly selected the original
  key from `.env` for this process, preserving `.env`, the second key and default
  client behavior. Sanitized evidence: `results/groq_quota_check.json`.
- Resumed under the existing conditional authorization. Completed the remaining
  synthetic evaluation: 876/876 gold assistant turns and 400/400 end-to-end
  conversations, up from 580 turns and 248 conversations. Independent coverage
  remains complete at 143/143 turns and 84/84 conversations; its prediction file
  is byte-for-byte unchanged. `results/eval_status.json` now records `complete`.
- Regenerated the complete S3 metrics, per-intent breakdowns, confusion matrices,
  baseline report and README table directly from saved predictions. Independent
  metrics include the frozen 1,000-resample conversation-cluster 95% intervals.
  S3 is GPT-OSS 120B through Groq; other systems have not run. The independent
  benchmark remains explicitly Claude-authored with Codex gold, not human evidence.
- Replayed all 484 saved end-to-end histories (84 independent, 400 synthetic),
  reproducing every tool result and recorded backend state. Previous checkpoint
  prefixes are intact, frozen inference hashes still match, and offline metric
  recomputation is identical. `results/phase3_resume_validation.json` contains
  final hashes and collection integrity evidence. No inference source, prompt,
  domain, dependency or decoding changes were made.
- `make lint` passed (67 Python files); `make test` passed all 232 tests in
  9.16 seconds. Phase 3 acceptance is complete. Stop here; Phase 4 remains
  unstarted and requires the user's explicit continuation. No GitHub push,
  dataset upload, GPU work or model download was performed.

## Phase 4: Kaggle notebooks prepared and locally validated (2026-10-16)

- User authorized Phase 4 only. Re-read the source spec and project rules. Created
  `notebooks/train_kaggle.ipynb` and `notebooks/eval_gpu_kaggle.ipynb`, with source
  generation in `scripts/build_notebooks.py` and schema/syntax validation in
  `scripts/validate_notebooks.py`. Both are output-free nbformat 4.5, with eight
  cells and five compiled code cells each. No GPU cell executed locally.
- Notebook bootstrap uses Kaggle Secrets, an isolated uv-managed Python 3.11
  environment and a user-pushed exact Git commit. Optional private GitHub access
  uses an environment-only authorization header; no credential is put in a URL,
  command argument, saved manifest or notebook output. HF account lookup can fill
  the dataset/model namespace, avoiding an extra required secret.
- Notebook pin set: PyTorch 2.6.0, Transformers 4.51.3, PEFT 0.15.2,
  TRL 0.17.0, Accelerate 1.6.0, bitsandbytes 0.45.5, datasets 3.5.0,
  W&B 0.19.10, safetensors 0.5.3, NumPy 1.26.4, tokenizers 0.21.4,
  sentencepiece 0.2.0, protobuf 5.29.4, tiktoken 0.9.0,
  opencv-python-headless 4.11.0.86, uv 0.11.26 and vLLM 0.8.5.post1.
  Verified all 17 release dates precede the two-week cutoff and checked 39
  declared requirements against selected direct pins. Versions, release dates,
  PyPI links and primary API evidence are in `results/kaggle_dependency_validation.json`.
  vLLM's pinned CUDA requirements match PyTorch 2.6.0; TRL's pinned source supports
  pretokenized datasets and `skip_prepare_dataset`. CUDA wheel/runtime compatibility
  is still unverified. No new local dependencies were installed or added to uv.lock.
- Added lazy, package-installable Kaggle workers under `cafe_agent.kaggle`.
  Training loads only the immutable raw train/validation splits, checks their
  content hashes/prompt/tools/authorship and reuses the shared assistant-mask
  implementation. Its collator preserves labels, including closing tokens and
  masked padding; TRL preprocessing/packing is disabled, and label_names is explicit.
- QLoRA defaults: NF4 double quantization, float16 (no bf16), seven attention/MLP
  projections, r=16/alpha=32/dropout=.05, lr=2e-4, three epochs, batch 4 and
  accumulation 4, cosine schedule, 3% warmup and gradient checkpointing. Probe
  the full 4,096-token padded batch in training mode before an optimizer step.
  OOM stops with an explicit batch-1/accumulation-16/new-run instruction, without
  truncation. Record probe peak separately from actual training/evaluation peak.
- Save/evaluate approximately every half epoch and export the best validation-loss
  adapter. Preserve immutable training inputs, installed versions, package commit,
  chosen train IDs, W&B link, losses and measured worker time. Checkpoint resumes
  require the matching full run manifest; completed adapter/summary pairs are reused.
- A separate Kaggle CPU process merges the adapter into a fresh fp16 base,
  builds llama.cpp b5046 at `74d4f5b041ad837153b0e90fc864b8290e01d8d5`
  with CUDA disabled and converts/quantizes Q4_K_M. Adapter, merged model and GGUF
  get private Hub repositories, model cards and actual training metadata. The
  summary prints immutable Hub revisions for the user's evaluation notebook.
- GPU evaluation runs S1/S2/S4 through the exact existing turn/end-to-end runner,
  parser, tools, clock and prediction schema. Uses float16, 8,192 context, seed 42,
  greedy decoding, 512 generated tokens and vLLM memory utilization .85. vLLM
  V0/XFormers settings target T4. Installation/load failure uses a fresh Transformers
  environment/process; runtime errors stop for resume. Backend files/manifests stay
  separate, with actual versions, GPU, model revision and source provenance recorded.
- S2 examples are frozen in `configs/few_shot.yaml`: first sorted train ID in four
  fixed intent groups (`train-00010`, `train-00004`, `train-00003`, `train-00000`).
  Their gold messages are retained and call IDs namespaced. No selection from test
  failures or prompt/evaluator tuning. Overflow is explicitly recorded as a length
  failure, never truncated. Bundle only complete S1/S2/S4 coverage, retaining S3.
- Restoring a saved Kaggle input validates all prediction/manifest files before
  mutation, preserves any newer current checkpoint prefix and rejects divergent
  histories. Training and GPU evaluation instructions cover interrupted sessions.
- To preserve reproducibility, kept `configs/train.yaml`, data-generation sources,
  S3 inference files, original manifests and predictions unchanged. New training
  settings/pins are in `configs/kaggle.yaml`, composed with the frozen tokenizer
  identity. The old Phase 1 model stubs remain frozen; notebook workers use new paths.
  Both existing data and S3 manifest checks still match their original fingerprints.
- Local acceptance: pinned raw data hashes passed; three real training examples
  loaded/rendered/masked/padded correctly (1,195/1,377/1,398 tokens, with
  19/113/118 assistant tokens). Checked all 1,019 S2 gold-history prompts; maximum
  length 3,269 plus 512 output tokens fits context 8,192. Evidence:
  `results/notebook_data_preflight.json` and `results/notebook_validation.json`.
- Final `make lint` passed (85 files, including the notebooks). `make test` passed
  all 251 tests in 4.73 seconds; 19 new cases cover data/authorship, deterministic
  subsets, assistant-label padding, half-epoch settings, few-shot context/IDs,
  raw/length failures, notebook regeneration, secret handling, fallback isolation,
  immutable manifests, checkpoint restoration and complete prediction bundling.
- Kaggle workers validate coverage/provenance without computing performance scores;
  the final metrics and confidence intervals remain a shared local scoring step.
- Offline source/wheel builds passed. The wheel imports every module outside the
  checkout without GPU packages and loads the bundled Kaggle/few-shot configs.
  Archives include notebooks/handoff docs and exclude secrets, caches, full generated
  data and model weights. Evidence: `results/phase4_integrity.json`. Diff checks passed.
- Added exact click-by-click instructions in `docs/PHASE4_KAGGLE.md` and linked them
  from README. Phase 4 implementation/acceptance is complete; actual T4 fit,
  training, GPU inference and GGUF export remain user-run work. No model loss,
  GPU-hour or S1/S2/S4 performance result was invented. No GitHub push, model
  upload, dataset upload, GPU install, training or later phase ran in this session.

## Phase 4 follow-up: newer W&B API keys (2026-10-20)

- User's Kaggle training worker stopped at `wandb.login` because W&B 0.19.10
  requires a 40-character key and their key has 86 characters. This happened
  after data encoding/manifest creation, before model loading or training steps.
  The generic key-in-code warning was not the failure; the key already enters
  the worker via the environment populated from Kaggle Secrets.
- Updated the notebook-only W&B pin to 0.22.3 in `configs/kaggle.yaml` and rebuilt
  both notebooks. The official release notes explicitly add support for keys
  longer than 40 characters:
  https://github.com/wandb/wandb/releases/tag/v0.22.3.
  PyPI reports release timestamp 2025-10-28T23:58:46.560250Z, older than the
  existing 2026-09-30 cutoff. Declared Python >=3.8, protobuf 5.29.4 and Pydantic
  2.11.7 requirements are compatible. Recorded metadata/checks in
  `results/kaggle_dependency_validation.json`; no local package was installed.
- Added an existing-session recovery procedure to `docs/PHASE4_KAGGLE.md`:
  update the W&B requirement and rerun bootstrap, choose a new run name/output
  directory, leave resume empty for this pre-training failure, then retry training.
  Keep the full key. Retain the failed attempt's immutable manifest rather than
  editing/deleting it. A GitHub package update is unnecessary for that pin-only
  Kaggle repair; this local commit is not pushed.
- Preserved the user's notebook GitHub settings and guide URL edits. Shortened
  the Git-ref comment to keep a populated SHA within the line limit. Notebook
  regeneration checks still compare every workflow cell, allowing customization
  of the designated first parameter cell. The original Phase 4 validation report
  remains historical evidence; this follow-up's notebook validation passed with
  eight cells/five compiled code cells each, no outputs and no GPU execution.
- Final `make lint` passed (85 files); `make test` passed all 251 tests in
  8.81 seconds. The initial lint check found only overlong comments beside the
  user's filled Git SHAs; shortened those comments, then reran checks successfully.
  No GPU dependencies/imports, training or W&B authentication ran on the Mac.
  Kaggle authentication with the updated SDK and actual training remain user-run
  verification. Stay in Phase 4; no Phase 5 work or GitHub push is authorized.

## Phase 4: user reports completed Kaggle training/export (2026-10-21)

- User ran the full training notebook after the batch-4 memory probe OOM, using
  the explicit batch-1/accumulation-16 fallback and a new run name,
  `main-r16-full-wandb0223-b1`. Their final output reports completed training,
  merge, GGUF conversion and all three private-by-workflow Hub uploads.
  Transcribed the supplied output into `results/kaggle_training_user_report.json`
  with explicit user-report provenance. Original Kaggle JSON files/manifests and
  remote artifact verification are still pending; no remote request was made.
- Reported train loss 0.025193520106762558; validation loss fell from
  1.008454442024231 to 0.010441209189593792 (also the best validation loss).
  Best checkpoint is checkpoint-330. These are assistant-token training/validation
  losses, not held-out action accuracy or task-success measurements.
- Reported Tesla T4 training time is 19,721.953115739 seconds (about 5h 28m).
  GPU-worker elapsed time is separately 5.591685450511111 hours, excluding setup
  and CPU export. Full-length probe shape [1, 4096] passed: peak allocated
  10,467,113,472 bytes (about 9.75 GiB); separately recorded training/evaluation
  peak 8,091,172,864 bytes (about 7.54 GiB). Do not merge the two measurements.
- Reported immutable Hub revisions:
  adapter `1727278e014684c60813ca7bbea6a145202731d7`;
  merged `6023f33114834618276c7de0cf43fda45397e1d7`;
  GGUF `a41f869ceec177228db7abdc0d01a07967a4f409`.
  Dataset/base/llama.cpp revisions match the existing configured pins. Full
  installed-package/Git provenance still requires the original training manifest.
- Filled the evaluation notebook's user parameter `MERGED_REVISION` with the
  reported merged revision; preserved the user's existing GitHub settings.
  Next: retain the full training output/checkpoints and original JSON files,
  run the separate GPU evaluation notebook with the same training Git reference,
  then download `gpu_predictions.zip`. GPU evaluation/S1-S2-S4 scores and local
  GGUF inference remain pending. No Phase 5 work, GPU code execution on the Mac,
  Hub visibility change or GitHub push was performed.
- Local `make lint` passed (85 files); `make test` passed all 251 tests in
  9.34 seconds. These verify the local handoff changes, not the uploaded models.

## Phase 4 follow-up: Kaggle download link returns 404 (2026-10-22)

- User reached the evaluation notebook's final download messages, but its link
  returned 404. The final cell used an absolute filesystem path in `FileLink`.
  Changed the generated notebook to retain the original validated archive under
  `cafe-gpu/`, copy it into `/kaggle/working/gpu_predictions.zip`, and display a
  relative filename link from the working directory. Print the copied archive's
  path/size and document the file-browser download fallback.
- Added a recovery cell to `docs/PHASE4_KAGGLE.md` that checks for the existing
  archive, copies it to the working-directory root and creates the relative link.
  It does not rerun model evaluation or rebundle predictions. Reference for
  relative FileLink usage:
  https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.FileLink.
- Rebuilt both notebooks while retaining the user's parameter cells. A local
  regression test executes the download cell with a fixture ZIP and mocked
  process/display interfaces, verifies that the link is relative and resolves
  to the copied archive, and checks that the original ZIP is unchanged/readable.
  No GPU notebook code or external service was executed locally.
- `make lint` passed (85 files); `make test` passed all 252 tests in 4.94 seconds.
  Both notebooks passed schema/syntax validation, with five compiled code cells
  and no outputs each. The user still needs to confirm the Kaggle browser
  download and provide the original prediction archive before local integrity
  checks/scoring. No GPU evaluation scores or Phase 5 results are claimed.

## Phase 4: GPU prediction receipt and integrity validation (2026-10-23)

- User supplied the extracted archive under `gpu_predictions/` and requested a
  check. Found six prediction JSONL files, three immutable manifests and the
  bundle summary. Every prediction/manifest SHA-256 matches the supplied summary;
  all three manifest fingerprints recompute exactly. Recorded actual checks,
  hashes, file sizes, coverage and provenance in
  `results/gpu_predictions_validation.json`.
- S1, S2 and S4 each have complete, unique schema-valid prediction keys:
  synthetic test 876 gold assistant turns plus 400 end-to-end conversations;
  independent_test 143 gold assistant turns plus 84 end-to-end conversations.
  Total 4,509 prediction records. Checked source IDs, author/intent labels,
  fingerprints and dataset hashes against the frozen local records, excluding
  Codex worked examples. Independent cases remain Claude-authored with Codex gold.
- All three systems actually used the Transformers fallback on Tesla T4, CUDA
  runtime 12.4, PyTorch 2.6.0 and Transformers 4.51.3. Do not label these as vLLM
  measurements. The bundle does not include attempts.json, so it does not establish
  the specific reason vLLM was unavailable. Dataset revision and S1/S2 base
  revision match configured pins; S4 uses the user's reported merged revision
  `6023f33114834618276c7de0cf43fda45397e1d7`. Package provenance records
  Git commit `3fb9663d21fd64a305a3ac9ec920518700c774cf`.
- All frozen inference and Kaggle worker source hashes match the local checkout.
  Verified prompt, tools, domain, runtime, decoding/context/seed and four fixed
  S2 examples against local configs. Relocated dataset paths only in memory for
  checking; retained the original manifests byte-for-byte.
- Replayed the recorded model responses through the shared agent runner on fresh
  backends, including explicit logical-time advances. All 1,452 end-to-end records
  reproduced their tool traces/results, final states, messages, errors and response
  consumption exactly. This validates execution integrity, not prediction accuracy.
  No new inference, model download, external API call or GPU import was performed.
- Copied all ten verified files into `results/predictions/` after checking every
  destination for conflicts. Copies match the supplied bytes. Preserved the
  original `gpu_predictions/` directory and every existing S3 file/hash.
  No metrics, README result table or performance reports were changed.
- `make lint` passed (85 files); `make test` passed all 252 tests in 4.99 seconds.
  Phase 4 GPU predictions are complete and ready for Phase 5. Original training
  JSON manifests/checkpoints remain separately requested; their absence here does
  not change prediction coverage. Stop pending `continue to Phase 5`; no GitHub
  push, Hub visibility change or Phase 5 work was performed.

## Known issues and user actions

- Benchmark preparation and private dataset upload are complete under the
  approved independent LLM-authored revision. Genuine human data is optional
  future work and must be labelled/reported separately. Keep the dataset private
  until the final publication checklist.
- The sandbox required permission for PyPI access and writes to `.git`.
- uv warns that its in-project cache may enter source distributions; archive
  inspection confirmed the ignore rules exclude it.
- The existing local `.env` must remain untracked and must not be printed.
- S3 evaluation is complete on both test sets, including independent-set
  confidence intervals. Keep free limits and preserve predictions/cache; future
  metrics recomputation and completed-run resumes require no new requests.
  The user reports completed GPU training/export; S1/S2/S4 GPU prediction files
  are now received and locally integrity-checked. Real local GGUF inference and
  the complete comparison/report remain pending Phase 5.
- The user reports T4 batch-1/4,096-token fit and completed training/GGUF export.
  Retain original JSON manifests, full checkpoints and private Hub artifacts.
  GPU evaluation has completed with the exact training Git reference and reported
  merged model revision. Retain the received prediction files and their backup;
  no GPU evaluation rerun is required.
- The account's 8,000-token/minute Groq limit slows generation. Parallel cache
  preparation hit that limit and stopped; the canonical resumable run completed
  with its existing bounded exponential backoff. Preserve cache/journals on retry.
- The interrupted pre-review drafts were preserved in the ignored
  `.cache/phase2-interrupted-drafts` directory before the finalized run.

## Next step

Phase 4 GPU predictions are received, complete, integrity-checked and copied
into results/predictions/ alongside unchanged S3 files. Preserve the original
training_summary.json, training_manifest.json, export_summary.json and full
checkpoint folders separately. Wait for `continue to Phase 5`
before downloading/evaluating GGUF or building the full report. Preserve all
existing predictions, immutable manifests and API cache. No GitHub remote or push
has been performed by Codex.
