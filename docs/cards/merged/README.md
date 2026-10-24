---
language: en
base_model: Qwen/Qwen2.5-1.5B-Instruct
license: apache-2.0
pipeline_tag: text-generation
tags:
- cafe
- tool-calling
- qlora
---

# PalakAnand30/cafe-agent-qwen2.5-1.5b-lora-merged

These cards are prepared locally for review; they have not been uploaded. Artifacts were private at the recorded handoff/download. No current visibility change or public deployment is implied.

| Artifact | Repository / immutable revision |
| --- | --- |
| adapter | [PalakAnand30/cafe-agent-qwen2.5-1.5b-lora](https://huggingface.co/PalakAnand30/cafe-agent-qwen2.5-1.5b-lora/tree/1727278e014684c60813ca7bbea6a145202731d7) · `1727278e014684c60813ca7bbea6a145202731d7` |
| merged | [PalakAnand30/cafe-agent-qwen2.5-1.5b-lora-merged](https://huggingface.co/PalakAnand30/cafe-agent-qwen2.5-1.5b-lora-merged/tree/6023f33114834618276c7de0cf43fda45397e1d7) · `6023f33114834618276c7de0cf43fda45397e1d7` |
| gguf | [PalakAnand30/cafe-agent-qwen2.5-1.5b-lora-gguf](https://huggingface.co/PalakAnand30/cafe-agent-qwen2.5-1.5b-lora-gguf/tree/a41f869ceec177228db7abdc0d01a07967a4f409) · `a41f869ceec177228db7abdc0d01a07967a4f409` |

Base: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Dataset: `PalakAnand30/cafe-tool-calls` at `5a626daa659b2455861e6a8d5a891c6dbe14e36d`. Training package commit: `3fb9663d21fd64a305a3ac9ec920518700c774cf`. Adapter and merged identities are from the user-supplied export receipt; the GGUF download additionally has independent LFS hash verification.

## Intended use

Research and local demonstration of structured café tool calls. Execute calls only through the project's validated tool schemas and fresh backend. Not validated for real customer ordering, payments, allergy advice or other domains. Menu shortcuts in the browser demo are serving conveniences, excluded from model benchmarks.

## Data and training

Synthetic splits: **3,500 train / 300 validation / 400 test**. Separate independent set: **84 Claude-authored conversations**, with Codex gold labels; **0 genuine human cases** and **5 excluded worked examples**.

Training used a Tesla T4, LoRA rank 16, 4,096-token full conversations, batch 1 × accumulation 16, 3 epochs and learning rate 0.0002. Measured GPU-worker time: 5.59 hours (not billing). Validation loss: 1.00845 → 0.01044; this is not test accuracy. Assistant payloads and closing tokens are supervised; user and tool content is masked. No conversation truncation.

[Training run](https://wandb.ai/palak-anand-the-university-of-edinburgh/cafe-tool-agent/runs/mz9t5xh9). Original training metadata and validation receipts are retained in the project's results directory.

LoRA alpha: 32; dropout: 0.05; targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Seed: 42; scheduler: cosine; optimizer: adamw_torch. The manifest's top-level batch/accumulation fields record the actual memory-safe run, overriding the nested trainer defaults.

Programmatic gold conversations were replayed against the café backend. Customer wording was paraphrased by `openai/gpt-oss-120b` through Groq with low reasoning effort, not through the OpenAI API. The independent customer text was supplied separately, authored by Claude and gold-labelled by Codex; it was not used for training or few-shot example selection.

## Evaluation

| System | Synthetic action + state | Independent action + state |
| --- | --- | --- |
| S1 | 29.0% | 26.2% [16.7%, 36.9%] |
| S2 | 32.5% | 21.4% [13.1%, 29.8%] |
| S3 | 57.5% | 40.5% [29.8%, 51.2%] |
| S4 | 91.5% | 33.3% [22.6%, 44.0%] |
| S5 | 91.8% | 31.0% [21.4%, 41.7%] |

The independent benchmark contains 84 Claude-authored cases with Codex gold labels, not human or real-customer evidence. It is weighted toward unsupported requests and has limited successful multi-item-order coverage. Synthetic templates and LLM wording can bias results. Intervals reflect case-sampling uncertainty, not real-world generalization. Plain-text actions use frozen approximate lexical rules; reply factuality and which detail was clarified are not judged. State-only success can overstate clarification, decline and lookup performance; action + state is the stronger endpoint. Hardware, quantization, tokenizer/runtime numerics and inference backends differ.

S1/S2/S4 ran with the recorded Transformers float16 fallback on Kaggle T4; the export bundle did not record why vLLM was unavailable. S3 used Groq. S5 used CPU-only llama.cpp Q4_K_M; hardware and precision differ. Independent brackets are conversation-cluster bootstrap intervals, not significance tests for paired differences.

## Reproduce

Use the project README and Phase 4–6 guides. Run `make setup`, `make setup-cpu`, `uv run --locked python scripts/download_models.py`, then `make serve` for CPU chat. An `HF_TOKEN` with read access is needed while repositories are private. GPU training and adapter/merged inference belong in the Kaggle notebooks; do not install GPU dependencies on the Mac. Match exact revisions and the shared chat renderer/parser; generic chat templates are not equivalent evidence.

## Licenses and attribution

The [Qwen base weights](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE) and [GPT-OSS paraphraser weights](https://huggingface.co/openai/gpt-oss-120b/blob/main/LICENSE) use Apache-2.0. Retain applicable upstream licenses, notices and modification attribution when distributing derivatives. Project code uses Apache-2.0; dataset content and rights held by the project owner use CC BY 4.0, including the owner-generated Claude benchmark. Attribution and source review are recorded in DATA_LICENSE.md and NOTICE. No GPT-OSS weights are distributed here.
