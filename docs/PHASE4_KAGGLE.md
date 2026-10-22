# Run the Phase 4 notebooks

Phase 4 prepares the notebooks. Training, T4 memory use, CUDA inference and GGUF
conversion have **not run locally**. The local preflight, syntax checks and tests
do not establish GPU compatibility or model quality. Keep the data/model Hub
repositories and Kaggle notebooks private until the final publication checklist.

## Push the exact project commit

1. On GitHub, click **New repository**. Name it `cafe-tool-agent`, choose **Public**,
   and leave the initial README, licence and `.gitignore` options unchecked.
2. In this project's terminal, run these commands, replacing `YOUR_USERNAME`.
   These are manual instructions; Codex has not pushed anything:

   ```sh
   git remote add origin https://github.com/YOUR_USERNAME/cafe-tool-agent.git
   git push -u origin main
   git rev-parse HEAD
   ```

3. Copy the complete 40-character commit printed by the last command. Use this
   exact value for `GIT_REF` in **both** notebooks. A branch such as `main` is not
   accepted, so later edits cannot silently change a run. If a remote already
   exists, use that repository's name instead of adding a second `origin`.
4. The notebooks also support a private GitHub repository: set `PRIVATE_GITHUB=True`
   and attach a `GITHUB_TOKEN` secret with read access. Tokens are passed through
   an environment-only Git authorization header, never embedded in URLs.

## Train on Kaggle

1. Open [Kaggle Code](https://www.kaggle.com/code), choose **New Notebook**, then
   **File → Import Notebook → Upload**. Select `notebooks/train_kaggle.ipynb`.
2. Open **Settings / Notebook options → Accelerator** and select **GPU T4 ×2**
   (or the available T4 option). The notebook exposes only GPU 0 to the worker;
   training uses one GPU. Turn **Internet** on. Complete Kaggle account verification
   if GPU/Internet controls are unavailable. See the
   [official notebook guide](https://www.kaggle.com/docs/notebooks) and
   [NVIDIA's Kaggle settings guide](https://docs.nvidia.com/datascience/deployment/26.10/platforms/kaggle/).
3. Open **Add-ons → Secrets**, create `HF_TOKEN` and `WANDB_API_KEY`, and enable
   access to this notebook for both. Use the existing HF account's token with
   access to the private dataset **and write permission for model repositories**.
   Use the same account/namespace that owns the uploaded dataset. Do not paste
   secret values into a code cell. Kaggle documents the mechanism in its
   [User Secrets announcement](https://www.kaggle.com/discussions/product-feedback/114053).
4. In the first code cell, set `GITHUB_USERNAME`, check `REPO_NAME`, and paste
   `GIT_REF`. Leave `HF_USERNAME=""` to derive your account name from the HF token.
   Defaults are `LORA_R=16`, `TRAIN_FRACTION=1.0`, `RUN_NAME="main-r16-full"`,
   `TRAIN_BATCH_SIZE=4`, and `GRAD_ACCUM=4`. Use a new run name for each ablation.
5. Choose **Save Version → Save & Run All** to run a retained notebook version.
   Follow its execution logs. For interactive debugging, run cells in order and
   download needed working files before ending that session. Bootstrap creates an isolated
   Python 3.11 environment in `/tmp`, so Kaggle's kernel Python version can differ.
   Pinned GPU packages are installed there; no local Mac setup is involved.
6. Check the first worker output: the pinned raw train/validation files must pass
   their hashes, and the three-example preflight must pass. The training worker
   masks only assistant payloads/closing tokens; its collator preserves those
   labels. No held-out test data is loaded by training.
7. The worker measures a forward/backward batch padded to 4,096 tokens before
   taking an optimizer step. If it reports OOM, set `TRAIN_BATCH_SIZE=1`,
   `GRAD_ACCUM=16` and a **new** `RUN_NAME`, then rerun. Do not shrink the token
   limit or truncate conversations. The probe excludes optimizer state; the
   separate training peak includes the actual training/evaluation allocation.
8. Follow the printed W&B run link. Training uses float16 NF4 QLoRA, gradient
   checkpointing, the configured three epochs and approximately half-epoch
   validation/save intervals. The best validation-loss adapter is saved.
9. Let the export cell finish. A separate CPU process reloads the original fp16
   base, merges the best adapter, builds the pinned llama.cpp without CUDA and
   converts/quantizes Q4_K_M. It uploads these **private** repositories:

   - `HF_USERNAME/cafe-agent-qwen2.5-1.5b-lora`
   - `HF_USERNAME/cafe-agent-qwen2.5-1.5b-lora-merged`
   - `HF_USERNAME/cafe-agent-qwen2.5-1.5b-lora-gguf`

10. The final cell prints the actual losses, worker runtime, GPU-worker hours,
    memory measurements, W&B URL, Hub URLs and exact Hub commit IDs. GPU-worker
    hours exclude notebook setup and CPU export and are not a billing receipt.
    Copy the merged repository's **40-character revision** for the evaluation notebook.
11. Open the completed saved version's **Output** tab. Reexecuting training in a
    session that retains its completed summary/adapter reuses them after checking
    the manifest; a fresh session has no such working files. Download `training_summary.json`,
    `training_manifest.json` and `export_summary.json` from the run's Output folder.

If training is interrupted, save the complete run output as a **private** Kaggle
input/dataset. Attach it to a new session, keep the same Git commit, parameters
and run name, and set `RESUME_CHECKPOINT` to its `checkpoints/checkpoint-N`
directory. Its grandparent must contain the matching `training_manifest.json`.
Resume restores optimizer/scheduler/RNG through Trainer; incompatible manifests
fail explicitly. Retain the checkpoints even if the notebook session expires.

### W&B rejects an 86-character key

`ValueError: API key must be 40 characters long, yours was 86` comes from the
old `wandb==0.19.10` pin. W&B added longer-key support in
[0.22.3](https://github.com/wandb/wandb/releases/tag/v0.22.3); the notebooks now
pin that version. Keep the complete key in Kaggle Secrets.

For a notebook already open on Kaggle:

1. In the bootstrap cell's `REQUIREMENTS`, replace `wandb==0.19.10` with
   `wandb==0.22.3`, then rerun that cell. This updates the isolated worker
   environment rather than Kaggle's notebook kernel.
2. Set `RUN_NAME = "main-r16-full-wandb0223"` in the parameters cell and rerun
   that cell. Keep `RESUME_CHECKPOINT = ""` for this failed pre-training attempt.
   The old attempt already wrote an immutable manifest containing installed
   versions; retain its folder and use a new output directory for the changed pin.
3. Rerun the training cell. This traceback occurred before loading model weights
   or taking training steps. The warning about specifying a key in code is
   generic: the worker obtains the value from its environment, populated by
   Kaggle Secrets. The warning itself does not stop training.

No GitHub package update is needed for this pin-only repair to an existing
Kaggle notebook. Future notebook regeneration uses `configs/kaggle.yaml`.

## Evaluate S1, S2 and S4

1. Create a **fresh** private Kaggle notebook and import
   `notebooks/eval_gpu_kaggle.ipynb` using the same File → Import Notebook flow.
2. Select T4 and Internet as above. Attach `HF_TOKEN`; W&B is not required for
   this notebook. Attach `GITHUB_TOKEN` only for a private GitHub repository.
3. Fill in the same GitHub fields and exact `GIT_REF`. Set `MERGED_REVISION` to
   the merged commit from training. The model ID is derived from `HF_USERNAME`.
4. Choose **Save Version → Save & Run All**. The pinned vLLM V0/XFormers path uses
   float16, context 8,192,
   one GPU, memory utilization 0.85, temperature 0, seed 42 and 512 output tokens.
   Installation or model-load failure switches to a fresh Transformers environment
   and process. Mid-run generation errors stop for resume; they do not mix backends
   in an existing prediction file. `attempts.json` and manifests identify actual backends.
5. Wait for S1, S2 and S4 to finish both modes on **400 synthetic** and **84 independent
   Claude-authored** conversations. S2 uses exactly four fixed training examples;
   S1/S4 use the shared zero-shot prompt. Every system reuses the local strict
   parser, real backend, explicit clock, scoring contract and append-only runner.
6. If interrupted, retain the output `predictions/` folder, including backend
   subdirectories, as a private Kaggle input. In a new session set
   `RESUME_PREDICTIONS` to that folder, preserving the Git/model revisions and
   settings. Manifest checks reject changed inputs or a mixed backend.
7. The final cell verifies complete prediction coverage and creates
   `/kaggle/working/cafe-gpu/gpu_predictions.zip`, then copies it to
   `/kaggle/working/gpu_predictions.zip` for a relative download link. Click that
   link, or download `gpu_predictions.zip` from the saved version's **Output**.
8. On the Mac, extract the ZIP's files into this project's `results/predictions/`.
   The archive contains S1/S2/S4 only; retain the existing S3 files. Keep the
   `gpu_evaluation_summary.json` beside the manifests for model/backend provenance.
9. Tell Codex the notebooks have run and the predictions are present. Phase 5
   will load GGUF on CPU and produce the shared local comparison/report. Phase 4
   itself does not claim completed GPU experiments or begin Phase 5.

If an older notebook's download link returns **404**, run this cell in the same
active evaluation session. It reuses the existing archive without running
evaluation or bundling again:

```python
import os
import shutil
from pathlib import Path

from IPython.display import FileLink, display

source = Path("/kaggle/working/cafe-gpu/gpu_predictions.zip")
if not source.is_file():
    raise FileNotFoundError("ZIP missing in this session; check the final evaluation cell.")
shutil.copyfile(source, "/kaggle/working/gpu_predictions.zip")
os.chdir("/kaggle/working")
display(FileLink("gpu_predictions.zip"))
```

If the link still fails, use the file browser's download action for
`/kaggle/working/gpu_predictions.zip`. An expired interactive session or an older
saved version may not contain the current run's files; preserve output before
ending the session.

## Local verification and implementation references

```sh
uv run --locked python scripts/build_notebooks.py
uv run --locked python scripts/validate_notebooks.py
uv run --locked python -m cafe_agent.kaggle.preflight --output results/notebook_data_preflight.json
make lint
make test
```

Local preflight needs the already generated ignored data files. CI unit tests
use small generated fixtures and do not require that corpus or GPU dependencies.
Do not execute either notebook's bootstrap/training/evaluation cells on the Mac.

`configs/train.yaml` retains the frozen model/tokenizer identity. New training
settings and exact notebook pins live in `configs/kaggle.yaml` so existing data
and S3 manifests remain valid. `configs/few_shot.yaml` records the first sorted
train ID per fixed intent, source hash and unchanged gold messages. Nothing was
selected from test failures. GPU workers live under `cafe_agent.kaggle`; the old
Phase 1 `models.kaggle` stubs remain frozen for S3 provenance and are not used.

Release dates, sources and declared dependency compatibility are recorded in
`results/kaggle_dependency_validation.json`. API choices follow the pinned
[TRL SFTTrainer source](https://github.com/huggingface/trl/blob/v0.17.0/trl/trainer/sft_trainer.py),
[vLLM CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.8.5.post1/requirements/cuda.txt)
and [llama.cpp conversion revision](https://github.com/ggml-org/llama.cpp/tree/74d4f5b041ad837153b0e90fc864b8290e01d8d5).
CUDA wheel installation, actual T4 fit, full training and GGUF conversion remain
unverified until these notebooks run on Kaggle. Report any failure with its cell,
error and saved artifacts; never share secret values.
