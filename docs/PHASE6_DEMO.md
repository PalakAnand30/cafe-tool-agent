# Run the café demo

The demo serves the exact Phase 5 GGUF on CPU, using the shared prompt, parser,
tool schemas, agent loop and in-process SQLite café service. Each customer has
an isolated café. Orders are simulated, expire with the session, and disappear
on restart. No money is charged and no real order is placed.

## Local browser

```sh
make setup
make setup-cpu
make serve
```

Open <http://127.0.0.1:7860>. The first startup downloads the pinned private GGUF
if it is missing; set `HF_TOKEN` in `.env`. Existing weights are verified against
the SHA-256 in `configs/serve.yaml`. Tokenizer files use the frozen base revision.
No evaluation dataset is downloaded for serving.

The redesigned counter uses the warm cream/peach/coffee palette from the supplied
Inspiration references, with original local SVG artwork, menu cards and a receipt
panel. It works without external font or image services. The menu tabs browse
current café data and show prices and out-of-stock items; swipe the cards on a phone.

Standalone menu requests such as “Show me the menu” and “show me the food menu”
now read the café tools directly. “Food” combines bakery and sandwiches; it is not
an added tool-schema category. Responses expose `source: "menu"` and actual tool
traces under **Behind the counter**. Orders and other chat still use the model
(`source: "model"`). This demo convenience does not change the model, frozen
evaluator or previously reported results; mixed requests still go to the model.

The five clickable examples cover ordering, clarification, declining an off-menu
request, recovering from an out-of-stock tool result, and menu browsing. Expand
**Behind the counter** to see actual calls/results. The receipt shows backend state,
including decimal-string totals. It is independent of the assistant's wording.
Use **Let 5 minutes pass** to progress preparation; checking status does
not advance time. Try ordering, asking for the returned order ID's status, then
cancelling it before it is ready. **Start fresh** clears that café; the chat's clear
button also resets the actual session rather than only hiding the transcript.

Sessions expire after one idle hour and allow 24 customer turns. History is never
silently truncated. At a context/output limit or interrupted model request, start
a new session. A request can execute a tool before generation fails; inspect the
order board before starting over. CPU inference is serialized across customers.
The application runs one worker; multiple workers would not share session state.

The default local host is loopback. `CAFE_HOST` and `PORT` override the configured
host and port. Do not use `uvicorn --reload` or multiple workers with the loaded
model. This demo has no accounts or durable order storage.

## Chat API

`GET /health` returns readiness after startup loads the model. `POST /chat`
accepts `{"message":"One medium latte, please.","session_id":null}`. Save its
returned `session_id` and include it in later requests. The response contains
`reply`, `trace`, `state`, `current_order`, and an optional `error`.

`POST /session/advance` and `POST /session/reset` take `{"session_id":"…"}`.
Reset returns HTTP 204. Expired/unknown sessions return 404; invalid requests
return 422; exhausted sessions require reset (409). Session IDs are random bearer
identifiers: keep them private to the browser. The UI keeps them in server-side
Gradio session state, rather than editable transcript text.

## Docker on Apple Silicon

Start Docker Desktop, then:

```sh
make docker-build
make docker-run
```

The image targets `linux/amd64`, exposes 7860, and runs as UID 1000. The run target
publishes only `127.0.0.1:7860` and mounts the existing `models/` and tokenizer
cache. If these directories do not exist, create them first. The `.env` file is
passed at runtime, excluded from the image and never uploaded. Stop with Ctrl-C.

The image installs the official **prebuilt CPU wheel** of llama-cpp-python 0.3.21,
released 2026-04-27. The official wheel index lacks the evaluated macOS 0.3.9
release. Platform-specific pins retain the Mac's CPU-only 0.3.9 build; Linux uses
0.3.21. Both disable GPU layers and K/Q/V offload, and startup rejects a build
that supports GPU offload. Container outputs are serving checks, not new S5
benchmark measurements. Emulation on Apple Silicon can be much slower than the
native Mac runtime. In the local check, the image loaded the model and served
the UI/health endpoint, but chat exceeded a 180-second client timeout under
amd64 emulation. Use native `make serve` for local chat; a live native-amd64 Space
chat check remains part of deployment verification. The lockfile still fixes every
dependency; the Docker wheel
install ignores the resolver's release-date filter because the official wheel
index omits upload dates, after its release date was independently verified.

## Prepare and deploy a Docker Space

Deployment is optional for local use. On 2026-10-24 the owner authorized a public
demo Space while keeping model and dataset repositories private. Use `--public`
only when that visibility is intended; the default deployment creates a private
Space and preserves existing visibility. The [publication guide](PHASE7_PUBLICATION.md)
records the release steps.

```sh
uv run --locked python scripts/deploy_space.py
```

This is a local dry run: inspect the printed `.cache/space-build/<hash>/` folder
and `results/space_deployment.json`. Only application source, configs, lockfile,
packaging files, Dockerfile and a Space README are included. No `.env`, model
weights, dataset, API cache, predictions, Git metadata or notebook is uploaded.

When ready to create/update `HF_USERNAME/cafe-tool-agent`, run:

```sh
uv run --locked python scripts/deploy_space.py --deploy
```

For a recruiter-accessible public demo, add `--public`:

```sh
uv run --locked python scripts/deploy_space.py --deploy --public
```

The token in `.env` must permit creating/updating that Space and reading the
private model. The script sets `HF_TOKEN`
as a Space secret through the SDK, and uploads the prepared bundle. Existing
Docker Spaces keep their visibility unless `--public` explicitly makes the demo
public; an existing different SDK causes an error.
It never changes model/dataset visibility, requests paid hardware, or pushes to
GitHub. Publication is a separate action; this phase only prepares the script.

After deployment, open the Space and check its build/runtime logs, health endpoint,
chat and tool trace. Startup downloads/cache-verifies weights. Default ephemeral
Space storage may require downloading them again after a restart; no paid storage
is configured. See [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
for runtime secrets, user permissions and storage behavior.

Record a short screen capture/GIF once the browser demo is running: order a
medium latte, expand the tool trace, then show the order board. Keep it as a
demo recording, not evidence of benchmark accuracy. Phase 7 handles final polish
and the publication checklist.
