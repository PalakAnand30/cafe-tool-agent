# Evidence-based CV bullets

- Built a shared tool-calling evaluation across five systems; fine-tuned S4 achieved 62.9% [52.8%, 71.8%] action accuracy versus prompted S2 46.2% [38.7%, 53.6%] on 84 independent Claude-authored cases with Codex gold (95% conversation bootstrap intervals).
- Ran Q4_K_M inference on local CPU with zero GPU layers; measured 31.0% [21.4%, 41.7%] action-and-state task success on the same LLM-authored benchmark.
- Produced reproducible schema validation, append-only prediction checkpoints, backend replay and deterministic reports on 400 synthetic and 84 independent cases per system; excluded worked examples and reported benchmark provenance limitations.
