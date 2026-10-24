# Reproducibility and publication checklist

Phase 7 provides documentation and a manual CPU regression workflow. On
2026-10-24 the owner authorized pushing GitHub, running Actions and deploying a
public demo Space, while retaining private model/dataset repositories. Project
code uses Apache-2.0 and owner-held dataset rights use CC BY 4.0; see LICENSE,
DATA_LICENSE.md and NOTICE. The owner confirmed the Claude benchmark was
generated in their own account. Actual remote outcomes are recorded in PROGRESS.md.

## Reproduce from a fresh checkout

Use Python 3.11 and uv 0.11.26. From the repository root:

```sh
make setup
make lint
make test
```

The tests require only tokenizer/config downloads, not weights or GPU packages.
Copy `.env.example` to `.env` only if an environment file does not already exist.
Set `HF_TOKEN` with read access to the private dataset and GGUF repository.

```sh
uv run --locked python scripts/download_models.py
make report
make release-docs
```

The downloader restores exact-revision raw test JSONL and checks frozen hashes.
Reporting uses committed predictions and does not call a model API. Documentation
generation reads `results/metrics.json`, training metadata and upload/download
receipts. The four prepared Hub cards live under `docs/cards/`; their hashes and
inputs are recorded in `results/publication_docs.json`. They are local copies,
not the current remote cards. Regenerate documentation after a new report.

For actual local chat or evaluation, follow [the CPU setup guide](PHASE5_LOCAL.md)
and [demo guide](PHASE6_DEMO.md). `make setup-cpu` compiles the forced CPU-only
build on the Mac. Normal `make setup`, lint or test can remove the optional CPU
extra; reinstall it before inference. The model downloader itself does not
require llama.cpp. GPU reproduction remains in [Kaggle](PHASE4_KAGGLE.md).

## Manual CPU regression guard

The workflow `.github/workflows/smoke_eval.yml` has only `workflow_dispatch`.
After the owner pushes it, add the repository Actions secret `HF_TOKEN` with
read access to the pinned private model/data, then run **CPU smoke evaluation**
from GitHub Actions. This uses GitHub-hosted CPU time; no paid hardware is
requested by the workflow. The token is passed as an environment secret, never
embedded in a URL or command.

The workflow installs locked dependencies and the pinned official Linux CPU
wheel, restores verified model/gold data, then runs **50 complete synthetic test
conversations** against fresh backends. The fixed seed samples case IDs without
reading outcomes. It uses the existing strict parser, agent, clock and scorer;
the browser menu helper is absent. No inference code is retuned from smoke
failures. Its stronger **action-and-state success** must be at least **0.80**.

Both settings live in the single `# smoke_eval: {...}` YAML comment pragma in
`configs/eval.yaml`. This explicit metadata preserves the frozen strict config
schema and the parsed configuration recorded in historical manifests. The smoke
loader rejects missing, duplicate or invalid metadata. Changing the threshold
or sample requires a new output directory and a documented decision. The guard
is an engineering regression check on synthetic data, not a generalization claim.

Run actual inference locally after CPU setup and model restoration:

```sh
make smoke-eval
```

Or verify the same sample against preserved S5 predictions without new inference:

```sh
uv run --locked python scripts/smoke_eval.py \
  --saved-dir results/predictions/cpu-v1 \
  --output-dir results/smoke-reference
```

The saved check is labelled `saved_predictions_check`; it is not execution of
the new workflow. Actual runs are labelled `cpu_inference`. Each output directory
contains `selection.json`, `summary.json` and, for inference, a CPU manifest and
append-only predictions. Completed cases are skipped on resume. Fingerprints
reject changed code/config/model inputs. Missing or duplicate cases fail rather
than being counted as complete. Failed thresholds exit nonzero. The Actions job
retains evidence on failure; download it before artifact retention expires.

The completed local Phase 7 run is archived in `results/smoke-phase7/`, with its
check receipt in `results/phase7_smoke_validation.json`. This archive is a copy
of the original resumable output, whose manifest records `results/smoke-cpu`.
It is separate regression evidence and is not merged into the five-system metrics.

Linux uses the pinned CPU wheel while the evaluated Mac run used its own pinned
native build. Numerical differences can change results. A failed guard should
be investigated with its trace, rather than lowering the threshold to pass.

## Before making anything public

1. Review the README, report, CV bullets and cards. Preserve benchmark authorship,
   coverage, confidence intervals, synthetic/independent differences, actual
   Transformers fallback and cost assumptions. Keep all original evidence.
2. Retain the selected Apache-2.0 code and CC BY 4.0 dataset licences and their
   notices. The owner confirmed the Claude benchmark's own-account origin;
   DATA_LICENSE.md records the source review and attribution scope. Keep the
   Claude-authored/Codex-labelled provenance intact.
3. Review model redistribution notices. The [Qwen base](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE)
   and [GPT-OSS paraphraser](https://huggingface.co/openai/gpt-oss-120b/blob/main/LICENSE)
   weights use Apache-2.0. Retain required license/notice files and modification
   attribution for any redistributed derivative. No GPT-OSS weights are bundled.
4. Review the exact Git changes and repository history for secrets and personal
   information. Keep `.env`, Kaggle secrets, raw API cache, weights and local
   databases excluded. The `Inspiration/` source images are not published assets;
   the demo uses original SVG/CSS artwork. Review saved predictions/screenshots
   and the W&B account link before public sharing.
5. Explicitly choose what to publish: code/report first; dataset, adapter, merged
   model, GGUF and Space separately. They can remain private. Changing a remote
   card creates a new Hub revision; keep the evaluated immutable revisions in
   provenance and do not relabel existing results as a new run.
6. Once authorized, push the reviewed Git commit and check push/PR CI plus the
   manual CPU workflow. Local passing checks do not establish that GitHub CI ran.
7. Optionally deploy the prepared private Space using the demo guide, verify
   build/runtime logs, `/health`, chat and tool traces. Docker startup/UI passed
   previously, but a complete chat under emulation timed out and remains a
   separate runtime check. The current authorization is for a public Space only;
   model and dataset repositories remain private.
8. Optionally record the demo: order a medium latte, expand the trace and show
   the board. Include the actual recording and label it as a demonstration.

## Completion boundary

Local checks and actual remote execution outcomes are recorded in `PROGRESS.md`.
Code/data licences and public Space publication were selected/authorized by the
owner on 2026-10-24. Deployment and CI success are established by their recorded
remote checks. A demo recording remains optional.
