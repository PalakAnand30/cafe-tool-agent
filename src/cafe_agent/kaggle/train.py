"""User-run Kaggle QLoRA job. GPU imports occur only inside main()."""

import argparse
import json
import math
import os
import time
from pathlib import Path

from cafe_agent.config import read_config
from cafe_agent.data.tokenization import load_tokenizer

from .checkpoint import prepare_run
from .data import AssistantCollator, encode_records, hub_records, pad_batch, select_train
from .provenance import installed_versions, package_provenance
from .settings import train_settings


def training_arguments(config: dict, count: int, batch: int, accumulation: int) -> dict:
    values = config["trainer"]
    half_epoch = max(1, math.ceil(math.ceil(math.ceil(count / batch) / accumulation) / 2))
    return {
        "num_train_epochs": values["epochs"],
        "learning_rate": values["learning_rate"],
        "warmup_ratio": values["warmup_ratio"],
        "lr_scheduler_type": values["lr_scheduler_type"],
        "optim": values["optim"],
        "per_device_train_batch_size": batch,
        "per_device_eval_batch_size": values["per_device_eval_batch_size"],
        "gradient_accumulation_steps": accumulation,
        "gradient_checkpointing": values["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "max_grad_norm": values["max_grad_norm"],
        "eval_strategy": "steps",
        "eval_steps": half_epoch,
        "save_strategy": "steps",
        "save_steps": half_epoch,
        "logging_steps": values["logging_steps"],
        "save_total_limit": values["save_total_limit"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "prediction_loss_only": True,
        "label_names": ["labels"],
        "fp16": True,
        "bf16": False,
        "tf32": False,
        "seed": config["seed"],
        "data_seed": config["seed"],
        "max_length": config["max_sequence_length"],
        "packing": False,
        "completion_only_loss": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "remove_unused_columns": False,
        "report_to": ["wandb"],
        "dataloader_num_workers": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.lora_r, args.batch_size, args.accumulation) <= 0:
        parser.error("Rank, batch size and accumulation must be positive")
    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one Kaggle T4 with CUDA_VISIBLE_DEVICES=0")
    config, kaggle = train_settings(), read_config("kaggle.yaml")
    job_started = time.perf_counter()
    token = os.environ["HF_TOKEN"]
    set_seed(config["seed"])
    _, train = hub_records(
        args.dataset_repo, kaggle["dataset_revision"], "train", kaggle["datasets"]["train"], token
    )
    _, validation = hub_records(
        args.dataset_repo,
        kaggle["dataset_revision"],
        "validation",
        kaggle["datasets"]["validation"],
        token,
    )
    train = select_train(train, args.train_fraction, config["seed"])
    tokenizer = load_tokenizer()
    tokenizer.padding_side = "right"
    encoded = encode_records(train, tokenizer, config["max_sequence_length"])
    valid = encode_records(validation, tokenizer, config["max_sequence_length"])
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train": config,
        "dataset_repo": args.dataset_repo,
        "dataset_revision": kaggle["dataset_revision"],
        "dataset_hashes": kaggle["datasets"],
        "train_ids": [row.id for row in train],
        "lora_r": args.lora_r,
        "batch_size": args.batch_size,
        "accumulation": args.accumulation,
        "run_name": args.run_name,
        "package": package_provenance(),
        "versions": installed_versions(),
    }
    summary_path = args.output / "training_summary.json"
    if prepare_run(args.output, manifest, args.resume):
        print(summary_path.read_text())
        return
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
    run = wandb.init(project=config["wandb_project"], name=args.run_name, config=manifest)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        token=token,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="sdpa",
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    model.train()
    # Forward/backward probe at the configured 4,096-token padded shape; no optimizer step.
    longest = sorted(encoded, key=lambda row: len(row["input_ids"]), reverse=True)
    probe = pad_batch(
        [longest[i % len(longest)] for i in range(args.batch_size)], tokenizer.pad_token_id
    )
    for index in range(args.batch_size):
        extra = config["max_sequence_length"] - len(probe["input_ids"][index])
        for name, fill in [
            ("input_ids", tokenizer.pad_token_id),
            ("attention_mask", 0),
            ("labels", -100),
        ]:
            probe[name][index] += [fill] * extra
    torch.cuda.reset_peak_memory_stats()
    try:
        tensors = {name: torch.tensor(value, device="cuda") for name, value in probe.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**tensors).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss during memory probe")
        loss.backward()
        torch.cuda.synchronize()
        probe_peak = torch.cuda.max_memory_allocated()
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            "4,096-token probe exceeded T4 memory. Set TRAIN_BATCH_SIZE=1 and "
            "GRAD_ACCUM=16 in a new run; retain full conversations and the token budget."
        ) from None
    model.zero_grad(set_to_none=True)
    del tensors, loss, probe
    torch.cuda.empty_cache()
    # Reset seed after the probe so its dropout does not change the training RNG stream.
    set_seed(config["seed"])
    options = training_arguments(config, len(encoded), args.batch_size, args.accumulation)
    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(args.output / "checkpoints"), run_name=args.run_name, **options
        ),
        train_dataset=Dataset.from_list(encoded),
        eval_dataset=Dataset.from_list(valid),
        data_collator=AssistantCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    initial = trainer.evaluate()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    final = trainer.evaluate()
    if not all(
        math.isfinite(value)
        for value in [result.training_loss, final["eval_loss"], trainer.state.best_metric]
    ):
        raise RuntimeError("Nonfinite training/validation loss; refusing to publish the run")
    adapter = args.output / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    summary = {
        "train_loss": result.training_loss,
        "initial_validation_loss": initial["eval_loss"],
        "validation_loss": final["eval_loss"],
        "best_validation_loss": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "runtime_seconds": seconds,
        "gpu_hours": (time.perf_counter() - job_started) / 3600,
        "gpu_hours_basis": (
            "Measured GPU-worker elapsed time; excludes notebook setup and export, "
            "not a billing receipt."
        ),
        "gpu": torch.cuda.get_device_name(0),
        "memory_probe_peak_bytes": probe_peak,
        "training_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "probe_shape": [args.batch_size, config["max_sequence_length"]],
        "memory_note": (
            "Probe excludes optimizer state; training peak includes training/evaluation."
        ),
        "wandb_url": run.url,
        "adapter_path": str(adapter),
        "dataset_revision": kaggle["dataset_revision"],
        "base_revision": config["revision"],
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    run.log(summary)
    run.finish()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
