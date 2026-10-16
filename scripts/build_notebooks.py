"""Build output-free, self-contained Kaggle notebooks using only local metadata."""

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from cafe_agent.config import read_config

RUFF_CONFIG = str(Path(__file__).resolve().parents[1] / "pyproject.toml")


def cell(kind: str, text: str) -> dict:
    source = textwrap.dedent(text).strip() + "\n"
    if kind == "code":
        source = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--config",
                RUFF_CONFIG,
                "--stdin-filename",
                "cell.py",
                "-",
            ],
            input=source,
            text=True,
        )
    value = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        compile(source, "<notebook cell>", "exec")
        value.update(execution_count=None, outputs=[])
    return value


def write(name: str, cells: list[dict]) -> None:
    value = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"isInternetEnabled": True, "accelerator": "nvidiaTeslaT4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path("notebooks", name).write_text(json.dumps(value, indent=1) + "\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--config",
            RUFF_CONFIG,
            str(Path("notebooks", name)),
        ],
        check=True,
        capture_output=True,
    )


def build() -> None:
    config = read_config("kaggle.yaml")
    dependencies = config["dependencies"]
    pins = [f"{name}=={version}" for name, version in dependencies.items() if name != "uv"]
    bootstrap = r"""
        import base64
        import json
        import os
        import re
        import subprocess
        import sys
        from pathlib import Path

        from kaggle_secrets import UserSecretsClient

        if not re.fullmatch(r"[a-f0-9]{40}", GIT_REF):
            raise ValueError("Set GIT_REF to the Phase 4 commit from Codex's report before running")
        if not re.fullmatch(r"[A-Za-z0-9-]+", GITHUB_USERNAME):
            raise ValueError("Fill in GITHUB_USERNAME")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", REPO_NAME):
            raise ValueError("Invalid repository name")
        secrets = UserSecretsClient()
        ENV = os.environ.copy()
        ENV["HF_TOKEN"] = secrets.get_secret("HF_TOKEN")
        if IS_TRAINING:
            ENV["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
        if PRIVATE_GITHUB:
            github_token = secrets.get_secret("GITHUB_TOKEN")
            credential = base64.b64encode(("x-access-token:" + github_token).encode()).decode()
            ENV.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
                       GIT_CONFIG_VALUE_0="AUTHORIZATION: basic " + credential)
            del github_token, credential
        ENV.update(CUDA_VISIBLE_DEVICES="0", TOKENIZERS_PARALLELISM="false", PYTHONHASHSEED="42",
                   HF_HOME="/tmp/cafe-hf", VLLM_USE_V1="0", VLLM_ATTENTION_BACKEND="XFORMERS",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8")
        # No credential appears in a command, URL, saved manifest or printed environment.
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
                        "uv==UV_VERSION"], check=True)
        UV = [sys.executable, "-m", "uv"]
        REQUIREMENTS = PINS_LITERAL
        PIN_FILE = Path("/tmp/cafe-notebook-requirements.txt")
        PIN_FILE.write_text("\n".join(REQUIREMENTS) + "\n")
        PACKAGE = f"git+https://github.com/{GITHUB_USERNAME}/{REPO_NAME}.git@{GIT_REF}"

        def make_environment(location):
            python = str(Path(location) / "bin/python")
            if not Path(python).is_file():
                subprocess.run(UV + ["venv", "--python", "3.11", str(location)],
                               check=True, env=ENV)
            subprocess.run(UV + ["pip", "install", "--python", python,
                           "--exclude-newer", "2026-09-30T00:00:00Z",
                           "-r", str(PIN_FILE), PACKAGE], check=True, env=ENV)
            return python

        PYTHON = make_environment("/tmp/cafe-notebook-env")
        if not HF_USERNAME:
            HF_USERNAME = subprocess.check_output([PYTHON, "-c",
                "import os; from huggingface_hub import HfApi; "
                "print(HfApi(token=os.environ['HF_TOKEN']).whoami()['name'])"],
                env=ENV, text=True).strip()
        DATASET_REPO = HF_USERNAME + "/cafe-tool-calls"
        print("Python 3.11 environment ready; credentials remain in process memory.")
    """.replace("UV_VERSION", dependencies["uv"]).replace("PINS_LITERAL", repr(pins))
    common = """
        GITHUB_USERNAME = ""  # Your GitHub username; no token here.
        REPO_NAME = "cafe-tool-agent"  # Match the repository you push.
        GIT_REF = "PASTE_PHASE4_COMMIT"  # Exact 40-character commit from the Phase 4 report.
        PRIVATE_GITHUB = False  # True requires an attached GITHUB_TOKEN Kaggle secret.
        HF_USERNAME = ""  # Blank: derive the account name from HF_TOKEN.
    """
    train_parameters = (
        common
        + """
        IS_TRAINING = True
        LORA_R = 16
        TRAIN_FRACTION = 1.0
        RUN_NAME = "main-r16-full"
        TRAIN_BATCH_SIZE = 4
        GRAD_ACCUM = 4
        RESUME_CHECKPOINT = ""  # Optional checkpoint directory restored from a prior saved run.
    """
    )
    train_run = """
        if not re.fullmatch(r"[A-Za-z0-9_-]+", RUN_NAME):
            raise ValueError("RUN_NAME must be a simple directory name")
        OUTPUT = Path("/kaggle/working") / ("cafe-" + RUN_NAME)
        OUTPUT.mkdir(exist_ok=True)
        subprocess.run([PYTHON, "-m", "cafe_agent.kaggle.preflight",
                        "--dataset-repo", DATASET_REPO, "--output", str(OUTPUT / "preflight.json")],
                       check=True, env=ENV)
        command = [PYTHON, "-m", "cafe_agent.kaggle.train", "--dataset-repo", DATASET_REPO,
                   "--output", str(OUTPUT), "--run-name", RUN_NAME,
                   "--lora-r", str(LORA_R), "--train-fraction", str(TRAIN_FRACTION),
                   "--batch-size", str(TRAIN_BATCH_SIZE), "--accumulation", str(GRAD_ACCUM)]
        if RESUME_CHECKPOINT:
            command += ["--resume", RESUME_CHECKPOINT]
        subprocess.run(command, check=True, env=ENV)
    """
    export = """
        # Training was a separate process and has released its GPU memory.
        # CPU-only merge, llama.cpp compilation/conversion, then private Hub uploads.
        subprocess.run([PYTHON, "-m", "cafe_agent.kaggle.export", "--output", str(OUTPUT),
                        "--namespace", HF_USERNAME, "--dataset-repo", DATASET_REPO],
                       check=True, env={**ENV, "CUDA_VISIBLE_DEVICES": ""})
    """
    summary = """
        summary = json.loads((OUTPUT / "export_summary.json").read_text())
        print(json.dumps(summary, indent=2))
        print("Copy this exact MERGED_REVISION into eval_gpu_kaggle.ipynb:")
        print(summary["hub"]["merged"]["revision"])
        print("Retain the saved version's Output and the W&B link.")
        print("For an interactive run, download needed working files before ending the session.")
    """
    write(
        "train_kaggle.ipynb",
        [
            cell(
                "markdown",
                """
        # Café QLoRA training — Kaggle T4

        User-run notebook; GPU training has not been tested locally. Set GPU T4 and Internet
        on. Attach HF_TOKEN (write access) and WANDB_API_KEY using Add-ons → Secrets.
        The pinned GitHub commit must be pushed before running. Keep this notebook and its
        output private; exported Hub repositories also start private. Optional private
        GitHub access uses GITHUB_TOKEN through an environment-only authorization header.

        Only train and validation are loaded here. No test result selects hyperparameters.
        Base model, dataset, dependencies and llama.cpp are pinned. All GPU imports live
        in the Kaggle Python 3.11 worker; the notebook kernel only orchestrates processes.
        """,
            ),
            cell("code", train_parameters),
            cell("code", bootstrap),
            cell(
                "markdown",
                """
        ## Train and measure T4 memory

        Uses NF4 double quantization, float16, all seven attention/MLP projections,
        assistant-only labels and gradient checkpointing. Default batch 4 × accumulation 4.
        A forward/backward memory probe uses the full 4,096-token padded shape before any
        optimizer step; its peak excludes optimizer state. The actual training peak is
        reported separately. If the probe OOMs, choose batch 1 / accumulation 16, give the
        run a new name and rerun; full conversations are retained. Do not lower the token
        limit or silently truncate. Evaluation/save occur approximately every half epoch;
        the best validation-loss checkpoint is exported. The W&B project is cafe-tool-agent.

        Checkpoints are resumable: save the Kaggle output, attach it as a private input in
        a new session and set RESUME_CHECKPOINT to its checkpoint directory. Keep the same
        Git commit, dataset, parameters and run name. Kaggle session/GPU quota may require
        more than one session; do not treat an interrupted training run as complete.
        """,
            ),
            cell("code", train_run),
            cell(
                "markdown",
                """
        ## Export and upload

        Merge the best adapter into a fresh fp16 base on CPU. Build the pinned llama.cpp
        with CUDA disabled; convert to f16 GGUF and quantize Q4_K_M. Upload adapter, merged
        fp16 and quantized model, each with a model card and actual training metadata.
        GPU-worker hours exclude notebook setup and CPU export and are not a billing receipt.
        """,
            ),
            cell("code", export),
            cell("code", summary),
        ],
    )
    eval_parameters = (
        common
        + """
        IS_TRAINING = False
        MERGED_REVISION = "PASTE_40_CHARACTER_MERGED_COMMIT"
        RESUME_PREDICTIONS = ""  # Optional prior /kaggle/input/... directory of backend folders.
    """
    )
    install_vllm = f"""
        result = subprocess.run(UV + ["pip", "install", "--python", PYTHON,
                    "--exclude-newer", "2026-09-30T00:00:00Z",
                    "--constraint", str(PIN_FILE), "vllm=={config["vllm_version"]}"], env=ENV)
        VLLM_READY = result.returncode == 0
        FALLBACK_PYTHON = None
        print("vLLM install:", "ready" if VLLM_READY else
              "failed; fresh Transformers fallback will be used")
    """
    evaluate = """
        if not re.fullmatch(r"[a-f0-9]{40}", MERGED_REVISION):
            raise ValueError("Copy the exact merged commit from the training export summary")
        OUTPUT = Path("/kaggle/working/cafe-gpu")
        OUTPUT.mkdir(exist_ok=True)
        if RESUME_PREDICTIONS:
            # Only use your own saved output from the exact same Git/model revisions.
            subprocess.run([PYTHON, "-c",
                "import sys; from pathlib import Path; "
                "from cafe_agent.kaggle.checkpoint import restore_predictions; "
                "restore_predictions(Path(sys.argv[1]), Path(sys.argv[2]))",
                RESUME_PREDICTIONS, str(OUTPUT / "predictions")], check=True, env=ENV)
        selection, attempts = {}, []
        for system in ["S1", "S2", "S4"]:
            backend = "vllm" if VLLM_READY else "transformers"
            while True:
                if backend == "transformers" and FALLBACK_PYTHON is None:
                    # A failed vLLM load cannot retain CUDA state in this new environment/process.
                    FALLBACK_PYTHON = make_environment("/tmp/cafe-transformers-env")
                python = PYTHON if backend == "vllm" else FALLBACK_PYTHON
                directory = OUTPUT / "predictions" / backend
                command = [python, "-m", "cafe_agent.kaggle.evaluate", "--system", system,
                           "--backend", backend, "--dataset-repo", DATASET_REPO,
                           "--merged-repo", HF_USERNAME + "/cafe-agent-qwen2.5-1.5b-lora-merged",
                           "--merged-revision", MERGED_REVISION, "--output", str(directory)]
                result = subprocess.run(command, env=ENV)
                attempts.append({"system": system, "backend": backend,
                                 "exit_code": result.returncode})
                (OUTPUT / "attempts.json").write_text(json.dumps(attempts, indent=2))
                if result.returncode == 42 and backend == "vllm":
                    VLLM_READY = False
                    backend = "transformers"
                    continue
                result.check_returncode()
                selection[system] = str(directory)
                break
        (OUTPUT / "selection.json").write_text(json.dumps(selection, indent=2))
    """
    bundle = """
        from IPython.display import FileLink, display

        subprocess.run([PYTHON, "-m", "cafe_agent.kaggle.bundle",
                        "--selection", str(OUTPUT / "selection.json"),
                        "--output", str(OUTPUT / "gpu_predictions.zip")], check=True, env=ENV)
        display(FileLink(str(OUTPUT / "gpu_predictions.zip")))
        print("In the saved version's Output, download cafe-gpu/gpu_predictions.zip.")
        print("Extract it into local results/predictions/, retaining existing S3 files.")
    """
    write(
        "eval_gpu_kaggle.ipynb",
        [
            cell(
                "markdown",
                """
        # Café GPU evaluation — S1, S2, S4

        Use a fresh private Kaggle T4 notebook with Internet and the HF_TOKEN secret.
        Fill in the same exact Git commit and the merged revision from training's final cell.
        Each system runs gold-history turns and complete real-agent conversations against
        fresh backends on synthetic test (400) and independent_test (84 Claude-authored cases
        with Codex gold). Worked examples are excluded. No Groq requests occur here.

        S1 and S4 use the exact shared prompt; S2 inserts four frozen train examples.
        Temperature 0, max_tokens 512, seed 42, float16, context 8,192, single T4,
        gpu_memory_utilization=0.85. vLLM V0/XFormers supports the T4 configuration.
        Installation/load failure triggers a fresh Transformers process; generation errors
        stop rather than mixing backends in one prediction file. Actual backends/revisions
        and provenance are recorded in manifests. These GPU paths remain unverified until run.
        """,
            ),
            cell("code", eval_parameters),
            cell("code", bootstrap),
            cell("code", install_vllm),
            cell(
                "markdown",
                """
        ## Run both evaluation modes

        All systems reuse the shared strict parser, tool executor, logical clock and
        append-only runner. Each model/backend has an immutable manifest and separate files.
        To resume an interrupted session, save output as a private Kaggle input and set
        RESUME_PREDICTIONS to the previous predictions directory containing backend folders.
        Preserve the same Git/model revisions and settings. Completed IDs are skipped.
        No history is truncated; context overflow is recorded as a length failure.
        """,
            ),
            cell("code", evaluate),
            cell(
                "markdown",
                """
        ## Download for local scoring

        The final cell checks complete coverage before bundling S1/S2/S4 predictions and
        manifests. Download the ZIP, extract into results/predictions/ alongside S3, and
        give the files back to Codex for Phase 5's shared local report. Do not overwrite
        the S3 files. If interrupted, keep the partial backend folders for a later resume.
        """,
            ),
            cell("code", bundle),
        ],
    )


if __name__ == "__main__":
    build()
