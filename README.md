# Café Ordering Agent

A reproducible project to fine-tune Qwen2.5-1.5B-Instruct with QLoRA for café
ordering tool calls, compare prompting and fine-tuning, and serve the resulting
GGUF model on CPU. `PROJECT_SPEC.md` defines the full scope and working rules.

Phase 1 provides the café domain, SQLite backend, tool schemas and executors,
strict tool-call parser, shared prompt helpers, model clients and agent loop.
Training, data generation, model evaluation and the chat demo have not run.
No model performance results are available.

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

- `src/cafe_agent/`: domain, backend, tools, parser, prompting, clients and
  agent loop; data, evaluation and demo modules remain placeholders.
- `configs/domain.yaml`: menu, prices, supported modifiers, stock and timing.
- `configs/runtime.yaml`: decoding, caching, retry, CPU and agent settings.
  Other YAML files remain placeholders for their planned phases.
- `tests/`: backend, tools, HTTP API, parser, agent and offline client tests.
- `notebooks/`, `scripts/`, `docker/`: placeholders for their planned phases.
- `data/`: generated data and API cache are ignored; human tests and small
  samples can be versioned.
- `results/`: versioned location for future predictions and reports.
- `.github/workflows/ci.yml`: locked setup, lint and tests on pushes and PRs.
- `AGENTS.md` and `PROGRESS.md`: working rules, decisions and phase status.

## Commands and phase boundaries

`make setup`, `make lint` and `make test` are implemented. Remaining targets
report the phase that will implement them and exit with an error:

| Target | Planned phase |
| --- | --- |
| `make data`, `make push-data` | 2 |
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

<!-- RESULTS:START -->
No experiments have run; results will be generated from `results/` in later phases.
<!-- RESULTS:END -->

Continue one phase at a time, following Section 0 of `PROJECT_SPEC.md`.
