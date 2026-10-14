# Generated dataset statistics

Measured from validated JSONL; no model evaluation.

Validated: 4200/4200 conversations.

| Split | Intent | Conversations | Mean messages | Mean tokens | Max tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| train | availability | 280 | 5.00 | 1255.85 | 1502 |
| train | cancel_order | 210 | 9.00 | 1382.74 | 1409 |
| train | clarification | 525 | 5.93 | 1289.82 | 1348 |
| train | menu | 210 | 5.00 | 1771.51 | 2816 |
| train | multi_order | 595 | 5.00 | 1333.19 | 1352 |
| train | order_status | 280 | 9.00 | 1384.95 | 1403 |
| train | out_of_scope | 210 | 3.00 | 1187.87 | 1200 |
| train | single_order | 630 | 5.00 | 1289.98 | 1305 |
| train | tool_error | 210 | 5.00 | 1385.23 | 1543 |
| train | unsupported | 350 | 3.00 | 1189.23 | 1203 |
| validation | availability | 24 | 5.00 | 1236.38 | 1245 |
| validation | cancel_order | 18 | 9.00 | 1381.61 | 1408 |
| validation | clarification | 45 | 5.93 | 1288.69 | 1339 |
| validation | menu | 18 | 5.00 | 1730.72 | 2808 |
| validation | multi_order | 51 | 5.00 | 1332.39 | 1346 |
| validation | order_status | 24 | 9.00 | 1385.71 | 1400 |
| validation | out_of_scope | 18 | 3.00 | 1187.61 | 1194 |
| validation | single_order | 54 | 5.00 | 1290.63 | 1299 |
| validation | tool_error | 18 | 5.00 | 1364.00 | 1543 |
| validation | unsupported | 30 | 3.00 | 1188.43 | 1199 |
| test | availability | 32 | 5.00 | 1247.69 | 1498 |
| test | cancel_order | 24 | 9.00 | 1382.83 | 1406 |
| test | clarification | 60 | 5.93 | 1289.95 | 1339 |
| test | menu | 24 | 5.00 | 1731.33 | 2808 |
| test | multi_order | 68 | 5.00 | 1334.31 | 1345 |
| test | order_status | 32 | 9.00 | 1385.44 | 1394 |
| test | out_of_scope | 24 | 3.00 | 1189.17 | 1196 |
| test | single_order | 72 | 5.00 | 1289.78 | 1299 |
| test | tool_error | 24 | 5.00 | 1376.04 | 1545 |
| test | unsupported | 40 | 3.00 | 1190.17 | 1199 |

Multi-turn conversations: 1050.
User turns: 5250; noisy turns: 801.
Paraphrase attempts: 1563; rejections: 178 (11.39%).

Rejection reasons: {"missing_order_line": 11, "missing_unsupported_request": 20, "unchanged": 147}

Leakage checks: {"held_out_combinations_observed": 3, "held_out_templates_observed": 10, "overlapping_user_texts": 0}
