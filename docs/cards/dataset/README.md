---
language: en
task_categories:
- text-generation
tags:
- synthetic
- tool-calling
configs:
- config_name: synthetic
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
- config_name: independent_test
  data_files:
  - split: test
    path: data/independent_test.jsonl
---

# PalakAnand30/cafe-tool-calls

Prepared locally; not uploaded. Frozen evaluated revision: `5a626daa659b2455861e6a8d5a891c6dbe14e36d`; the upload receipt records private visibility. No visibility change is made.

## Construction and provenance

Synthetic splits: **3,500 train / 300 validation / 400 test**. Separate independent set: **84 Claude-authored conversations**, with Codex gold labels; **0 genuine human cases** and **5 excluded worked examples**.

Training used a Tesla T4, LoRA rank 16, 4,096-token full conversations, batch 1 × accumulation 16, 3 epochs and learning rate 0.0002. Measured GPU-worker time: 5.59 hours (not billing). Validation loss: 1.00845 → 0.01044; this is not test accuracy. Assistant payloads and closing tokens are supervised; user and tool content is masked. No conversation truncation.

[Training run](https://wandb.ai/palak-anand-the-university-of-edinburgh/cafe-tool-agent/runs/mz9t5xh9). Original training metadata and validation receipts are retained in the project's results directory.

Training/test synthetic template families and item/modifier pairs were reserved before sampling. Programmatic assistant calls, results and final states are schema-validated and replayed; normalized customer-text leakage is checked across synthetic splits. Groq served `openai/gpt-oss-120b` at low reasoning effort for customer paraphrases. Raw API responses are cached locally and are not part of the scored splits. Claude supplied the independent customer messages; Codex annotated domain-compatible gold calls and recorded overrides. The path `human_test` is historical: this is not a human-written benchmark. Legacy `author=codex_example` records are excluded from all scores and are not advertised as a dataset configuration.

## Format

Raw JSONL follows the project's `Conversation` schema: ID, split, author, intent, messages, tool schemas, action annotations, explicit clock advances and expected final state. Tool arguments are dictionaries in the gold data, converted to native strings by the shared renderer. Money remains decimal strings. Optional argument omission is meaningful: use raw JSONL rather than Arrow conversion that can insert null fields.

## Uses and limitations

Research on this configured menu/backend only. The independent benchmark contains 84 Claude-authored cases with Codex gold labels, not human or real-customer evidence. It is weighted toward unsupported requests and has limited successful multi-item-order coverage. Synthetic templates and LLM wording can bias results. Intervals reflect case-sampling uncertainty, not real-world generalization. Plain-text actions use frozen approximate lexical rules; reply factuality and which detail was clarified are not judged. State-only success can overstate clarification, decline and lookup performance; action + state is the stronger endpoint. Hardware, quantization, tokenizer/runtime numerics and inference backends differ.

Do not train on either test set or tune prompts/evaluator from observed failures. Keep customer authorship separate from gold annotation provenance. Publication requires the owner's rights review for supplied material.

## File integrity

| File | SHA-256 |
| --- | --- |
| `data/human_test.jsonl` | `2d647ef19500409343d48ca2110add1a0c5e00ded7f0495df2408ff3c5f7a19d` |
| `data/independent_test.jsonl` | `344b3436285d0864d4c56a7601ba2430bb425d344ec24c473e18455262d2fe73` |
| `data/test.jsonl` | `662e3d6a609b850ea7a2b36756edddd950df716ec2f9c8a7ad3d9ef5d2746754` |
| `data/train.jsonl` | `1326a0dc1c8db4cee0f4a745e86ad9af505c7f562c9b99cdea2cafb24e3bbf84` |
| `data/validation.jsonl` | `c83e6b580bf18b91b88c84b11eb965aae00db4aaa7d9512619d546465f08dfa8` |

## License status

A dataset redistribution license has not been selected. Do not infer permission for the dataset from an upstream model's Apache-2.0 license. [GPT-OSS weights license](https://huggingface.co/openai/gpt-oss-120b/blob/main/LICENSE); [Qwen weights license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE). Confirm rights and attribution for the supplied Claude material before release.
