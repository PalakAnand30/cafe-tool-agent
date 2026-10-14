# S3 baseline

| System | Set | Coverage (turn / end) | Action accuracy | Argument exact | Field F1 | End success | Action + state |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S3 | independent_test (**partial**) | 33/143 turns; 24/84 conversations | 30.3% | 0.0% | 0.0% | 75.0% | 8.3% |
| S3 | test (**partial**) | 0/876 turns; 0/400 conversations | — | — | — | — | — |

Bracketed intervals are 95% conversation-cluster bootstrap intervals on the complete independent LLM-authored set. Partial runs have no confidence intervals. Action/argument scores use gold-history turns; end scores use the real agent.

Final-state success can overstate clarification/decline/lookup performance because these tasks may leave the backend empty. The stricter action + state column also checks user-turn actions and missing/extra tool calls. Text actions use frozen lexical rules and do not verify reply factuality, tone, or which detail was clarified.

Customer text in independent_test was authored by Claude and labelled by Codex. The set is weighted toward unsupported requests and has limited successful multi-item-order coverage. These results are not human or real-customer evidence.

metrics.json contains field scores, precision/recall, per-intent breakdowns, confusion matrices, uncached request latency (including network), cost assumptions, coverage and provenance. Cached requests are excluded from latency.

Paid-equivalent prices checked 2026-10-14: [Groq model pricing](https://console.groq.com/docs/model/openai/gpt-oss-120b). This estimate excludes input-cache discounts and is not an actual bill. S1, S2, S4 and S5 have not been evaluated in Phase 3.
