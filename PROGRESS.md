# Project progress

Updated: 2026-10-14 (Europe/London).

## Phase checklist

- [x] Phase 0: scaffold.
- [ ] Phase 1: domain, backend, tools, parser and agent loop.
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

## Known issues and user actions

- No blocking issues. No accounts, uploads or secret values are required for Phase 0.
- The sandbox required permission for PyPI access and writes to `.git`.
- uv warns that its in-project cache may enter source distributions; archive
  inspection confirmed the ignore rules exclude it.
- The existing local `.env` must remain untracked and must not be printed.

## Next step

Stop after the Phase 0 local commit. Start Phase 1 only after the user says
`continue to Phase 1`. No remote has been added and nothing has been pushed.
