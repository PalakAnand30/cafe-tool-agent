# CPU evaluation and deterministic reporting

Phase 5 runs the exported Q4_K_M model on the local CPU and scores S1–S5 with
the existing shared parser, backend, action rules and metrics. Keep Hub model
and dataset repositories private. Nothing here pushes to GitHub or changes
Hub visibility.

## Prepare the locked CPU environment

From the project root, with Python 3.11 and uv:

```sh
make setup
make setup-cpu
uv run --locked python scripts/download_models.py
```

Normal setup does not install the CPU extra. The explicit CPU setup compiles
the existing `llama-cpp-python==0.3.9` pin with GPU backends disabled. This pin's
[upstream CMake file](https://github.com/abetlen/llama-cpp-python/blob/v0.3.9/CMakeLists.txt)
forces Metal on macOS, so ordinary `GGML_METAL=OFF` is insufficient. Our checked-in
`configs/cpu_only.cmake` is included at each nested CMake project, overriding
that setting. The setup script copies the hook to a temporary path to handle
project paths containing spaces. A working C/C++ compiler is required.

The CPU manifest checks that the native build has no GPU-offload support and
records native library hashes. Both `n_gpu_layers=0` and `offload_kqv=False` are
explicit. No CUDA torch, bitsandbytes, vLLM or MLX is installed or imported locally.
If normal setup later removes the CPU extra, rerun `make setup-cpu` before
inference; do not substitute an unchecked Metal wheel.

The downloader reads `HF_TOKEN` from `.env`, downloads the exact revision in
`configs/local.yaml`, and verifies the GGUF against the Hub's LFS SHA-256 and
file size. Weights stay in the ignored `models/` folder. Original training
metadata is downloaded from the same immutable model revision. No token appears
in command arguments, output, receipts or Git files.

The same command restores missing raw gold test files from the pinned private
dataset revision on a fresh checkout, verifying their frozen hashes. Existing
files are verified and never overwritten with different data. This avoids
regenerating the corpus just to reproduce scoring; keep the independent benchmark
source/override files that are already committed.

## Evaluate and benchmark

```sh
make eval-local
make latency
```

`make eval-local` now evaluates S5. The historical Groq path is available as
`make eval-s3`; do not rerun it to build the report. S1/S2/S4 predictions come
from Kaggle and remain immutable.

S5 uses the same full prompt and schemas, a frozen tokenizer, context 8,192,
temperature 0, seed 42 and 512 output tokens. It verifies native GGUF prompt
token IDs against the shared tokenizer before generation. It sets repetition
penalty to 1.0 to match unpenalized greedy GPU generation. Context overflow is
recorded as a length failure; conversations are never truncated or repaired.

Predictions append under the configured `results/predictions/cpu-v1/`. Both
gold-history assistant turns and full conversations run against fresh backends
with the original scripted customer messages and explicit clock advances.
Re-running skips completed IDs. Keep the manifest, prediction files and any
interrupted-tail backups. Changed inference inputs or code require a new output
directory (`python -m cafe_agent.cpu.evaluate --output-dir ...`); also point the
local report config to the selected run before reporting.

Run the latency benchmark after evaluation, without another inference workload
running concurrently. It uses three excluded warm-up requests and 100 fixed,
seeded synthetic gold-history prompts. Each request resets the KV context,
then measures rendering, tokenizer checks, prefill and generation. Model load,
warm-up and explicit reset are excluded. It records prompt/output lengths,
response hashes and per-request times in append-only JSONL. A matching manifest
is required to resume. Evaluation request latencies may reuse prompt prefixes;
the cold-context benchmark is reported separately.

## Rebuild the report offline

```sh
make report
# Equivalent full default report:
uv run --locked python scripts/evaluate.py --metrics-only
```

The full report requires complete S1/S2/S3/S4/S5 coverage and 100 unique latency
samples. Missing or partial inputs fail explicitly. It does not perform model
inference or contact any API. Historical S3 `--config`/`--output-dir` workflows
remain available through the original evaluator CLI.

Outputs include:

- `results/metrics.json`: shared metrics, per-intent scores, confusion matrices,
  coverage, model/backend provenance and conversation-cluster confidence intervals.
- `results/report.md`: comparison, latency, cost and limitations.
- `results/charts/`: standalone deterministic PNG/SVG comparison bars, intent
  heatmaps, confusion matrices and latency-versus-accuracy scatter.
- `results/error_analysis.json` and `.md`: deterministic categories and ten
  S4/S2 failure comparisons, without prompt or evaluator tuning.
- `results/cv_bullets.md` and the README results block: values read from the
  persisted metrics artifact, with benchmark authorship and limits explicit.
- `results/report_validation.json`: prediction, manifest and output hashes.

Independent customer text was authored by Claude and labelled by Codex; it is
not human evidence. Intervals use 1,000 conversation-cluster bootstrap resamples,
seed 42. Frozen lexical rules approximate clarification/decline actions and do
not verify reply factuality. Report state-only and action-and-state task success
separately. GPU systems used their recorded Transformers backend on T4; S3
includes network time, and S5 uses local CPU, so latency differences mix hardware
and runtimes.

S3 cost uses the dated saved paid token rates, not an actual bill. S5's CPU-hour
price is an explicit illustrative assumption in `configs/local.yaml`, excluding
hardware purchase, setup and training. S1/S2/S4 GPU cost is unknown without a
GPU-hour price; zero API fees do not imply free compute.

Keep all predictions, model revisions, training metadata and checkpoints.
Phase 6 (demo/Docker/deployment) requires a separate continuation.
