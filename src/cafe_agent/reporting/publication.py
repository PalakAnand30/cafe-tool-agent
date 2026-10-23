"""Prepare reviewable documentation from saved evidence, without publishing anything."""

import hashlib
import json
from pathlib import Path

from .text import LIMITATIONS, percent

QWEN_LICENSE = (
    "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/"
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE"
)
GPT_OSS_LICENSE = "https://huggingface.co/openai/gpt-oss-120b/blob/main/LICENSE"


def replace_block(text: str, name: str, body: str) -> str:
    start, stop = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    if text.count(start) != 1 or text.count(stop) != 1:
        raise ValueError(f"Expected exactly one {name} marker pair")
    prefix, remainder = text.split(start)
    _, suffix = remainder.split(stop)
    return prefix + start + "\n" + body.strip() + "\n" + stop + suffix


def render_publication(root: Path = Path(".")) -> dict[str, str]:
    """Read persisted results; numeric claims have no independent hand-written copies."""
    result = root / "results"
    data = json.loads((result / "metrics.json").read_text())
    if not data["all_complete"] or any(not group["complete"] for group in data["groups"]):
        raise ValueError("Publication documentation requires complete saved results")
    upload = json.loads((result / "dataset_upload.json").read_text())
    receipt = json.loads((result / "model_download.json").read_text())
    report = json.loads((result / "kaggle_training_user_report.json").read_text())
    artifacts = report["reported_export_summary"]["hub"]
    training = data["training"]
    manifest, summary = training["manifest"], training["summary"]
    groups = {(group["system"], group["split"]): group for group in data["groups"]}

    def metric(system: str, split: str, name: str, mode: str = "end_to_end") -> dict:
        return groups[(system, split)]["modes"][mode]["metrics"][name]

    independent = groups[("S5", "independent_test")]["expected_conversations"]
    synthetic = groups[("S5", "test")]["expected_conversations"]
    headline = (
        "A café ordering agent with a validated SQLite tool backend, a QLoRA-trained "
        "Qwen2.5-1.5B model and a CPU chat demo. The exported Q4_K_M model achieved "
        f"**{percent(metric('S5', 'test', 'action_and_state_success'))}** action-and-state "
        f"success on {synthetic} synthetic cases and "
        f"**{percent(metric('S5', 'independent_test', 'action_and_state_success'))}** "
        f"on {independent} independent Claude-authored cases with Codex gold labels. "
        f"The larger GPT-OSS model through Groq reached "
        f"**{percent(metric('S3', 'independent_test', 'action_and_state_success'))}** "
        "on the independent set. These are LLM-authored benchmarks, not customer evidence."
    )
    counts = manifest["dataset_hashes"]
    facts = (
        f"Synthetic splits: **{counts['train']['conversations']:,} train / "
        f"{counts['validation']['conversations']:,} validation / "
        f"{counts['test']['conversations']:,} test**. Separate independent set: "
        f"**{independent} Claude-authored conversations**, with Codex gold labels; "
        f"**{upload['human_conversations']} genuine human cases** and "
        f"**{upload['excluded_codex_examples']} excluded worked examples**.\n\n"
        f"Training used a {summary['gpu']}, LoRA rank {manifest['lora_r']}, "
        f"{manifest['train']['max_sequence_length']:,}-token full conversations, "
        f"batch {manifest['batch_size']} × accumulation {manifest['accumulation']}, "
        f"{manifest['train']['trainer']['epochs']} epochs and learning rate "
        f"{manifest['train']['trainer']['learning_rate']}. Measured GPU-worker time: "
        f"{summary['gpu_hours']:.2f} hours (not billing). Validation loss: "
        f"{summary['initial_validation_loss']:.5f} → {summary['validation_loss']:.5f}; "
        "this is not test accuracy. Assistant payloads and closing tokens are supervised; "
        "user and tool content is masked. No conversation truncation.\n\n"
        f"[Training run]({summary['wandb_url']}) · "
        "[Original metadata](results/training/training_manifest.json) · "
        "[Dataset validation](results/independent_validation.json)"
    )
    outputs = {}
    card_facts = facts.split("\n\n[Training run]")[0] + (
        f"\n\n[Training run]({summary['wandb_url']}). Original training metadata and "
        "validation receipts are retained in the project's results directory."
    )
    readme = (root / "README.md").read_text()
    readme = replace_block(readme, "HEADLINE", headline)
    outputs["README.md"] = replace_block(readme, "DATA_TRAINING", facts)
    table = [
        "| System | Synthetic action + state | Independent action + state |",
        "| --- | --- | --- |",
    ]
    for system in ["S1", "S2", "S3", "S4", "S5"]:
        table.append(
            f"| {system} | {percent(metric(system, 'test', 'action_and_state_success'))} | "
            f"{percent(metric(system, 'independent_test', 'action_and_state_success'))} |"
        )
    evidence = "\n".join(table) + "\n\n" + LIMITATIONS
    identity = ["| Artifact | Repository / immutable revision |", "| --- | --- |"]
    for kind, artifact in artifacts.items():
        identity.append(
            f"| {kind} | [{artifact['repo']}](https://huggingface.co/"
            f"{artifact['repo']}/tree/{artifact['revision']}) · "
            f"`{artifact['revision']}` |"
        )
    common = (
        "These cards are prepared locally for review; they have not been uploaded. "
        "Artifacts were private at the recorded handoff/download. No current visibility "
        "change or public deployment is implied.\n\n" + "\n".join(identity) + "\n\n"
        f"Base: `{manifest['train']['model_id']}` at `{manifest['train']['revision']}`. "
        f"Dataset: `{manifest['dataset_repo']}` at `{manifest['dataset_revision']}`. "
        f"Training package commit: `{manifest['package']['git_commit']}`. "
        "Adapter and merged identities are from the user-supplied export receipt; "
        "the GGUF download additionally has independent LFS hash verification.\n\n"
        "## Intended use\n\nResearch and local demonstration of structured café tool calls. "
        "Execute calls only through the project's validated tool schemas and fresh backend. "
        "Not validated for real customer ordering, payments, allergy advice or other domains. "
        "Menu shortcuts in the browser demo are serving conveniences, "
        "excluded from model benchmarks.\n\n"
        "## Data and training\n\n"
        + card_facts
        + f"\n\nLoRA alpha: {manifest['train']['lora']['alpha']}; "
        f"dropout: {manifest['train']['lora']['dropout']}; targets: "
        + ", ".join(f"`{name}`" for name in manifest["train"]["lora"]["target_modules"])
        + f". Seed: {manifest['train']['seed']}; "
        f"scheduler: {manifest['train']['trainer']['lr_scheduler_type']}; "
        f"optimizer: {manifest['train']['trainer']['optim']}. "
        "The manifest's top-level batch/accumulation fields record the actual "
        "memory-safe run, overriding the nested trainer defaults."
        + "\n\nProgrammatic gold conversations were replayed against the café backend. "
        "Customer wording was paraphrased by `openai/gpt-oss-120b` through Groq with "
        "low reasoning effort, not through the OpenAI API. The independent customer text "
        "was supplied separately, authored by Claude and gold-labelled by Codex; "
        "it was not used for training or few-shot example selection.\n\n"
        "## Evaluation\n\n" + evidence + "\n\n"
        "S1/S2/S4 ran with the recorded Transformers float16 fallback on Kaggle T4; "
        "the export bundle did not record why vLLM was unavailable. S3 used Groq. "
        "S5 used CPU-only llama.cpp Q4_K_M; hardware and precision differ. "
        "Independent brackets are conversation-cluster bootstrap intervals, not "
        "significance tests for paired differences.\n\n"
        "## Reproduce\n\nUse the project README and Phase 4–6 guides. "
        "Run `make setup`, `make setup-cpu`, "
        "`uv run --locked python scripts/download_models.py`, then `make serve` for CPU chat. "
        "An `HF_TOKEN` with read access is needed while repositories are private. "
        "GPU training and adapter/merged inference belong in the Kaggle notebooks; "
        "do not install GPU dependencies on the Mac. Match exact revisions and the shared "
        "chat renderer/parser; generic chat templates are not equivalent evidence.\n\n"
        "## Licenses and attribution\n\n"
        f"The [Qwen base weights]({QWEN_LICENSE}) and "
        f"[GPT-OSS paraphraser weights]({GPT_OSS_LICENSE}) use Apache-2.0. "
        "Retain applicable upstream licenses, notices and modification attribution when "
        "distributing derivatives. This does not establish a license for the supplied "
        "Claude-authored benchmark or this project's code/data. Those owner decisions "
        "remain on the publication checklist. No GPT-OSS weights are distributed here.\n"
    )
    for kind, artifact in artifacts.items():
        metadata = (
            "---\nlanguage: en\nbase_model: "
            + manifest["train"]["model_id"]
            + "\npipeline_tag: text-generation\ntags:\n- cafe\n- tool-calling\n- qlora\n"
            + ("- gguf\n" if kind == "gguf" else "")
            + "---\n\n"
        )
        extra = ""
        if kind == "gguf":
            extra = (
                f"File: `{receipt['filename']}` · {receipt['bytes']:,} bytes.\n\n"
                f"Verified SHA-256: `{receipt['sha256']}`. "
                "Inference uses `n_gpu_layers=0` and `offload_kqv=False`. "
                "The evaluated Mac build uses llama-cpp-python 0.3.9; the Docker/Linux "
                "CPU wheel is 0.3.21 and its full benchmark has not been rerun.\n\n"
            )
        outputs[f"docs/cards/{kind}/README.md"] = (
            metadata + f"# {artifact['repo']}\n\n" + extra + common
        )
    dataset_card = (
        "---\nlanguage: en\ntask_categories:\n- text-generation\ntags:\n- synthetic\n"
        "- tool-calling\nconfigs:\n- config_name: synthetic\n  data_files:\n"
        "  - split: train\n    path: data/train.jsonl\n"
        "  - split: validation\n    path: data/validation.jsonl\n"
        "  - split: test\n    path: data/test.jsonl\n"
        "- config_name: independent_test\n  data_files:\n"
        "  - split: test\n    path: data/independent_test.jsonl\n---\n\n"
        f"# {manifest['dataset_repo']}\n\nPrepared locally; not uploaded. "
        f"Frozen evaluated revision: `{upload['revision']}`; the upload receipt records "
        "private visibility. No visibility change is made.\n\n"
        "## Construction and provenance\n\n" + card_facts + "\n\n"
        "Training/test synthetic template families and item/modifier pairs were reserved "
        "before sampling. Programmatic assistant calls, results and final states are "
        "schema-validated and replayed; normalized customer-text leakage is checked across "
        "synthetic splits. Groq served `openai/gpt-oss-120b` at low reasoning effort for "
        "customer paraphrases. Raw API responses are cached locally and are not part of "
        "the scored splits. Claude supplied the independent customer messages; Codex "
        "annotated domain-compatible gold calls and recorded overrides. The path "
        "`human_test` is historical: this is not a human-written benchmark. "
        "Legacy `author=codex_example` records are excluded from all scores and are not "
        "advertised as a dataset configuration.\n\n"
        "## Format\n\nRaw JSONL follows the project's `Conversation` schema: ID, split, "
        "author, intent, messages, tool schemas, action annotations, explicit clock "
        "advances and expected final state. Tool arguments are dictionaries in the "
        "gold data, converted to native strings by the shared renderer. Money remains "
        "decimal strings. Optional argument omission is meaningful: use raw JSONL "
        "rather than Arrow conversion that can insert null fields.\n\n"
        "## Uses and limitations\n\nResearch on this configured menu/backend only. "
        + LIMITATIONS
        + "\n\nDo not train on either test set or tune prompts/evaluator "
        "from observed failures. Keep customer authorship separate from gold annotation "
        "provenance. Publication requires the owner's rights review for supplied material.\n\n"
        "## File integrity\n\n| File | SHA-256 |\n| --- | --- |\n"
        + "\n".join(
            f"| `{name}` | `{entry['sha256']}` |"
            for name, entry in upload["verified_files"].items()
            if name.endswith(".jsonl")
        )
        + "\n\n## License status\n\nA dataset redistribution license has not been selected. "
        "Do not infer permission for the dataset from an upstream model's Apache-2.0 "
        f"license. [GPT-OSS weights license]({GPT_OSS_LICENSE}); "
        f"[Qwen weights license]({QWEN_LICENSE}). "
        "Confirm rights and attribution for the supplied Claude material before release.\n"
    )
    outputs["docs/cards/dataset/README.md"] = dataset_card
    return outputs


def write_publication(root: Path = Path(".")) -> None:
    outputs = render_publication(root)
    for name, content in outputs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    inputs = [
        "metrics.json",
        "dataset_upload.json",
        "model_download.json",
        "kaggle_training_user_report.json",
    ]
    receipt = {
        "published": False,
        "inputs": {
            name: hashlib.sha256((root / "results" / name).read_bytes()).hexdigest()
            for name in inputs
        },
        "outputs": {
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in outputs.items()
        },
        "status": "Local review copies; code/data licenses, GIF, deployment and remote CI pending.",
    }
    (root / "results/publication_docs.json").write_text(json.dumps(receipt, indent=2) + "\n")
