# Independent benchmark provenance

The user approved replacing the original human-written benchmark requirement
with the supplied 84 Claude-authored cases, retaining the current café domain.

- `human_test_authoring.yaml` preserves the supplied customer messages and
  original gold suggestions unchanged. Its comments describe the original,
  superseded domain assumptions; it is not the scoring file.
- `gold_overrides.yaml` contains Codex-reviewed current-domain actions, revised
  intents, reasons, and source/domain hashes. No customer message is rewritten.
- `independent_test.jsonl` is the scoring file, produced with `make benchmark`.
  It uses `split=independent_test`, `author=claude`, and annotation provenance
  identifying Codex. Tool results and final states come from fresh backend replay.
- `human_test.jsonl` contains the five legacy Codex worked examples. They remain
  excluded from scoring. Genuine human cases may be added later with the CLI
  and must be reported separately.

No orders are initially seeded. Unsupported items/modifiers are declined before
calling tools; missing sizes and ambiguous croissant types trigger clarification.
Status/cancellation queries for uncreated IDs return `ORDER_NOT_FOUND`.

The old expected actions targeted another menu. Keeping the existing domain means
many cases now test declines, with limited successful multi-item-order coverage.
This is an independently LLM-authored set with Codex annotations, not a human-written
benchmark. See `results/independent_validation.json` for measured coverage and
`results/human_authoring_audit.json` for the initial compatibility audit.
