# Café Ordering Agent

A reproducible project to fine-tune Qwen2.5-1.5B-Instruct with QLoRA for café
ordering tool calls, compare prompting and fine-tuning, and serve the resulting
GGUF model on CPU. `PROJECT_SPEC.md` defines the full scope and working rules.

The café backend and agent core are implemented, together with the Phase 2
dataset pipeline, replay/token validation, assistant-only labels, human-test
annotation CLI and private dataset publisher. Training, model evaluation and
the chat demo have not run. No model performance results are available.

## Local setup

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
make setup
make lint
make test
```

The src-layout distribution is named `cafe-tool-agent`; its Python import is
`cafe_agent`. The Python version and direct dependency versions are pinned;
`uv.lock` records the resolved environment. Local setup installs no GPU libraries.

When a later phase needs credentials, copy `.env.example` to `.env` only if
you do not already have one, then fill in the required values locally.
Never commit or share `.env`.

## Project layout

- `src/cafe_agent/`: domain, backend, tools, parser, prompting, clients,
  agent loop and dataset pipeline; evaluation and demo remain placeholders.
- `configs/domain.yaml`: menu, prices, supported modifiers, stock and timing.
- `configs/runtime.yaml`: decoding, caching, retry, CPU and agent settings.
  `data.yaml` controls generation; `train.yaml` pins the tokenizer/model revision.
- `tests/`: backend, tools, HTTP API, parser, agent, client, replay, leakage,
  masking, annotation and publishing tests.
- `scripts/`: human-test annotation and private dataset upload entry points.
  `notebooks/` and `docker/` remain placeholders.
- `data/`: generated data and API cache are ignored; human tests and small
  samples can be versioned.
- `results/`: versioned location for future predictions and reports.
- `.github/workflows/ci.yml`: locked setup, lint and tests on pushes and PRs.
- `AGENTS.md` and `PROGRESS.md`: working rules, decisions and phase status.

## Commands and phase boundaries

`make setup`, `make lint`, `make test`, `make data` and `make push-data` are
implemented. Remaining targets report their planned phase and exit with an error:

| Target | Planned phase |
| --- | --- |
| `make eval-local` | 3 (harness), 5 (GGUF) |
| `make report` | 5 |
| `make serve`, `make docker-build`, `make docker-run` | 6 |

GPU training and inference run only on Kaggle T4 notebooks. Local GGUF
inference will use CPU only (`n_gpu_layers=0`).

## Café backend

Run the standalone café API from the repository root:

```sh
uv run --locked uvicorn cafe_agent.backend.app:create_app --factory
```

Visit `http://127.0.0.1:8000/docs` for the routes. The service persists orders
in the SQLite path configured in `configs/runtime.yaml`. This is the café
backend; the `/chat` endpoint and Gradio demo arrive in Phase 6.

The API exposes `GET /health`, `GET /menu`, `POST /availability`,
`POST /orders`, `GET /orders/{order_id}`, `POST /orders/{order_id}/cancel`
and `POST /tools/{name}`. The tool route and in-process executor share exactly
the same Pydantic validation and structured domain errors. Money is represented
as decimal strings in GBP on the wire to preserve exact totals.

For replay and evaluation, create a fresh in-memory backend:

```python
from cafe_agent.backend.service import CafeService

cafe = CafeService()
try:
    result = cafe.call("create_order", {
        "items": [{"item_name": "latte", "size": "medium", "quantity": 1}]
    })
    cafe.advance_time(1)
    state = cafe.snapshot()
finally:
    cafe.close()
```

Order progression uses a persisted logical clock. `advance_time(minutes)`
changes it explicitly; status checks and elapsed wall-clock time do not.
Cancellation is allowed while received or preparing, and repeated cancellation
is idempotent. Seeded preparation estimates and IDs replay deterministically.

## Model clients and agent loop

`models.base.ModelClient` provides a shared message and response interface.
`GroqClient` uses native tool calling, reads `GROQ_API_KEY` through python-dotenv
on the first cache miss, caches raw responses by request hash, and retries
rate limits and server errors with exponential backoff and jitter.

`LlamaCppClient` lazy-loads the optional `cpu` extra, fixes `n_gpu_layers=0`,
and accepts a tokenizer so it can use the shared Qwen chat-template renderer.
The optional engine has not been installed or tested with a real GGUF yet;
that validation belongs to Phase 5. HF and vLLM clients are explicit Kaggle
stubs for Phase 4 and import no GPU libraries locally.

`Agent.run(messages)` accepts caller-owned history, validates every generated
call before execution, appends tool results with matching IDs, and returns a
reply, history and trace. It stops on malformed output, reused call IDs or the
configured round limit. Model behaviour is tested with scripted responses and
mock transports; no live model quality claim is made.

## Generate and validate the dataset

Set `GROQ_API_KEY` in your existing `.env`, then run:

```sh
make data
```

The user-approved paraphraser and later S3 baseline are `openai/gpt-oss-120b`
through Groq with low reasoning effort. The originally specified Llama model
was unavailable to the configured account. This project does not use the OpenAI
API for these requests.

The job writes append-only gold drafts and final train/validation/test JSONL
under `data/generated/`, caches every successful Groq response in `data/cache/`,
and resumes completed IDs and paraphrase batches. A manifest rejects changed
inputs or code; use a new `--output-dir` when intentionally making a new version.
Do not delete the cache to retry an interrupted job.

Only customer utterances are paraphrased. Protected entities and complete order
lines are verified; rejected paraphrases fall back to the original utterance.
Assistant tool calls, backend results and varied replies are programmatic.
Reserved template families and item/modifier pairs belong exclusively to test,
and normalized customer text is deduplicated across all synthetic splits.

Every conversation validates its tool calls, replays stored results and final
state against a fresh backend, and fits the pinned Qwen template under the
user-approved full-conversation token budget. Assistant-token masking uses
template-aligned character spans and fast-tokenizer offsets; it never truncates
or labels customer/tool text as an assistant target. See
`data/generated/stats.md` for measured counts, lengths and rejection rates.

Regenerate the versioned review sample and validation evidence from those files:

```sh
uv run --locked python scripts/collect_data_artifacts.py
```

The resulting `data/samples/` and `results/data_validation.json` contain gold
data validation evidence, not model evaluation results.

An explicit offline variant is available for development. It records that no
API paraphrasing ran and uses a separate output directory:

```sh
uv run --locked python -m cafe_agent.data.generator --no-paraphrase --output-dir data/offline
```

## Write the human benchmark, then publish privately

Before seeing any model evaluation results, write your own conversations:

```sh
uv run --locked python scripts/annotate_human_test.py
```

Choose an intent, enter customer messages, and label each assistant action as
`tool`, `clarify`, `decline` or `reply`. Tool actions accept a function name and
JSON arguments, execute against a fresh backend, and let you annotate the next
assistant action. Use `/advance MINUTES` to test order progression, `/save` to
review/accept or type the expected final state, and `/quit` to exit. Saving
replays the entire conversation and checks the token budget. Previously saved
records survive interruption.

The file `data/human_test/human_test.jsonl` contains clearly marked
`author=codex_example` worked examples. They are excluded by the shared human
loader and must never count as human benchmark evidence. Add the human-authored
benchmark required by `configs/data.yaml`; do not generate it with an LLM.

Then set `HF_TOKEN` (dataset write permission) and `HF_USERNAME` in `.env` and run:

```sh
uv run --locked python scripts/push_dataset.py --dry-run
make push-data
```

The dry run stages validated files without contacting the Hub. Upload requires
the configured human benchmark count and creates a private
`<HF_USERNAME>/cafe-tool-calls` dataset. Synthetic splits use its default config;
human test is a separate config. Existing public repositories are rejected.
Read raw JSONL files from the Hub to preserve omitted argument fields; Arrow
inference can fill missing nested keys with null. Make the dataset public only
when the project is ready.

<!-- RESULTS:START -->
No experiments have run; results will be generated from `results/` in later phases.
<!-- RESULTS:END -->

Continue one phase at a time, following Section 0 of `PROJECT_SPEC.md`.
