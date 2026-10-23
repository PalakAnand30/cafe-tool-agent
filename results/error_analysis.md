# S4 versus S2 failure analysis

Frozen parser/action rules; no reply-factuality judge. Independent text is Claude-authored with Codex gold. Examples are illustrative, not prevalence estimates.

```json
{
  "categories": {
    "S2": {
      "invalid_output": 9,
      "success": 656,
      "wrong_action": 309,
      "wrong_arguments": 45
    },
    "S4": {
      "invalid_output": 5,
      "success": 915,
      "wrong_action": 78,
      "wrong_arguments": 21
    }
  },
  "end_to_end_categories": {
    "S2": {
      "PARSE_ERROR": 9,
      "required_action_failure": 149,
      "success": 148,
      "wrong_final_state": 178
    },
    "S4": {
      "PARSE_ERROR": 5,
      "required_action_failure": 52,
      "success": 394,
      "wrong_final_state": 33
    }
  },
  "failure_turns": 373,
  "scope": "Category counts pool both sets; per-set metrics are reported separately.",
  "selection": "First sorted distinct category/category/intent, then first remaining."
}
```

## Example 1: independent_test/h001

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "wrong_arguments",
    "S4": "wrong_arguments"
  },
  "conversation_id": "h001",
  "customer_messages": [
    "Can I get a large latte with oat milk please?"
  ],
  "expected_arguments": [
    {
      "arguments": {
        "items": [
          {
            "item_name": "latte",
            "modifiers": [
              "oat_milk"
            ],
            "quantity": 1,
            "size": "large"
          }
        ]
      },
      "id": "call_2",
      "name": "create_order"
    }
  ],
  "expected_label": "tool:create_order",
  "intent": "single_order",
  "responses": {
    "S2": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"Alex Turner\", \"items\": [{\"item_name\": \"latte\", \"quantity\": 1, \"modifiers\": [\"oat_milk\"], \"size\": \"large\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 56,
        "prompt_tokens": 1727,
        "total_tokens": 1783
      }
    },
    "S4": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"John Smith\", \"items\": [{\"item_name\": \"latte\", \"quantity\": 2, \"modifiers\": [], \"size\": \"large\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 51,
        "prompt_tokens": 1169,
        "total_tokens": 1220
      }
    }
  },
  "split": "independent_test"
}
```

## Example 2: independent_test/h002

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "wrong_action",
    "S4": "wrong_action"
  },
  "conversation_id": "h002",
  "customer_messages": [
    "medium cappuccino pls"
  ],
  "expected_arguments": [
    {
      "arguments": {
        "items": [
          {
            "item_name": "cappuccino",
            "modifiers": [],
            "quantity": 1,
            "size": "medium"
          }
        ]
      },
      "id": "call_2",
      "name": "create_order"
    }
  ],
  "expected_label": "tool:create_order",
  "intent": "single_order",
  "responses": {
    "S2": {
      "cached": false,
      "content": "Got it. I'll place a medium cappuccino for Alex Turner. Total: £2.50. ETA: 6 minutes.",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 31,
        "prompt_tokens": 1721,
        "total_tokens": 1752
      }
    },
    "S4": {
      "cached": false,
      "content": "How many would you like?",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 7,
        "prompt_tokens": 1163,
        "total_tokens": 1170
      }
    }
  },
  "split": "independent_test"
}
```

## Example 3: independent_test/h003

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "wrong_action",
    "S4": null
  },
  "conversation_id": "h003",
  "customer_messages": [
    "One flat white, extra shot, thanks"
  ],
  "expected_arguments": [],
  "expected_label": "clarify",
  "intent": "clarification",
  "responses": {
    "S2": {
      "cached": false,
      "content": "Flat white with extra shot: ORD-00001. Total: £2.50. Preparation time: 6 minutes.",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 30,
        "prompt_tokens": 1723,
        "total_tokens": 1753
      }
    },
    "S4": {
      "cached": false,
      "content": "Which size would you like: small, medium or large?",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 13,
        "prompt_tokens": 1165,
        "total_tokens": 1178
      }
    }
  },
  "split": "independent_test"
}
```

## Example 4: independent_test/h004

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": null,
    "S4": "wrong_action"
  },
  "conversation_id": "h004",
  "customer_messages": [
    "Could I have a small earl grey with lemon?"
  ],
  "expected_arguments": [],
  "expected_label": "decline",
  "intent": "unsupported",
  "responses": {
    "S2": {
      "cached": false,
      "content": "I'm sorry, but Earl Grey tea does not support the 'lemon' modifier. Would you like a small Earl Grey tea without lemon?",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 30,
        "prompt_tokens": 1726,
        "total_tokens": 1756
      }
    },
    "S4": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"John Doe\", \"items\": [{\"item_name\": \"earl grey tea\", \"quantity\": 1, \"modifiers\": [], \"size\": \"small\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 53,
        "prompt_tokens": 1168,
        "total_tokens": 1221
      }
    }
  },
  "split": "independent_test"
}
```

## Example 5: independent_test/h005

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "invalid_output",
    "S4": "invalid_output"
  },
  "conversation_id": "h005",
  "customer_messages": [
    "2 croissants, warmed up if possible"
  ],
  "expected_arguments": [],
  "expected_label": "clarify",
  "intent": "clarification",
  "responses": {
    "S2": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"Alex Turner\", \"items\": [{\"item_name\": \"butter croissant\", \"quantity\": 1, \"modifiers\": [], \"size\": \"small\"}, {\"item_name\": \"almond croissant\", \"quantity\": 1, \"modifiers\": [], \"size\": \"small\"}]}}\n</tool_call>\n<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"Alex Turner\", \"items\": [{\"item_name\": \"butter croissant\", \"quantity\": 1, \"modifiers\": [], \"size\": \"small\", \"warm\": \"w\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 138,
        "prompt_tokens": 1724,
        "total_tokens": 1862
      }
    },
    "S4": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"John Doe\", \"items\": [{\"item_name\": \"butter croissant\", \"quantity\": 2, \"modifiers\": [], \"size\": \"warmed\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 54,
        "prompt_tokens": 1166,
        "total_tokens": 1220
      }
    }
  },
  "split": "independent_test"
}
```

## Example 6: independent_test/h007

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "wrong_action",
    "S4": "invalid_output"
  },
  "conversation_id": "h007",
  "customer_messages": [
    "I'd like a big iced latte w/ caramel syrup"
  ],
  "expected_arguments": [],
  "expected_label": "decline",
  "intent": "unsupported",
  "responses": {
    "S2": {
      "cached": false,
      "content": "I'm sorry, but we don't have a big iced latte. Would you like a large iced latte with caramel syrup instead?",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 31,
        "prompt_tokens": 1728,
        "total_tokens": 1759
      }
    },
    "S4": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"John Doe\", \"items\": [{\"item_name\": \"iced latte\", \"quantity\": 1, \"modifiers\": [], \"size\": \"big\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 52,
        "prompt_tokens": 1170,
        "total_tokens": 1222
      }
    }
  },
  "split": "independent_test"
}
```

## Example 7: independent_test/h009

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "wrong_action",
    "S4": "wrong_arguments"
  },
  "conversation_id": "h009",
  "customer_messages": [
    "Can I grab a large decaf mocha? Name's Priya"
  ],
  "expected_arguments": [
    {
      "arguments": {
        "customer_name": "Priya",
        "items": [
          {
            "item_name": "mocha",
            "modifiers": [
              "decaf"
            ],
            "quantity": 1,
            "size": "large"
          }
        ]
      },
      "id": "call_2",
      "name": "create_order"
    }
  ],
  "expected_label": "tool:create_order",
  "intent": "single_order",
  "responses": {
    "S2": {
      "cached": false,
      "content": "Got it. I'll place a large decaf mocha for Priya.",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 17,
        "prompt_tokens": 1729,
        "total_tokens": 1746
      }
    },
    "S4": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"Priya\", \"items\": [{\"item_name\": \"decaf mocha\", \"quantity\": 1, \"modifiers\": [], \"size\": \"large\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 53,
        "prompt_tokens": 1171,
        "total_tokens": 1224
      }
    }
  },
  "split": "independent_test"
}
```

## Example 8: independent_test/h010

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "wrong_action",
    "S4": "wrong_action"
  },
  "conversation_id": "h010",
  "customer_messages": [
    "three small green teas with honey"
  ],
  "expected_arguments": [],
  "expected_label": "decline",
  "intent": "unsupported",
  "responses": {
    "S2": {
      "cached": false,
      "content": "I'm sorry, but the menu does not include green tea with honey. Could you please specify another drink?",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 23,
        "prompt_tokens": 1721,
        "total_tokens": 1744
      }
    },
    "S4": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"John Doe\", \"items\": [{\"item_name\": \"green tea\", \"quantity\": 3, \"modifiers\": [\"honey\"], \"size\": \"small\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 54,
        "prompt_tokens": 1163,
        "total_tokens": 1217
      }
    }
  },
  "split": "independent_test"
}
```

## Example 9: independent_test/h011

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "wrong_action",
    "S4": null
  },
  "conversation_id": "h011",
  "customer_messages": [
    "ham and cheese toastie on gluten free bread please"
  ],
  "expected_arguments": [],
  "expected_label": "decline",
  "intent": "unsupported",
  "responses": {
    "S2": {
      "cached": false,
      "content": "I'm sorry, but we don't have a gluten-free ham and cheese toastie on the menu. Could you please specify another item?",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 29,
        "prompt_tokens": 1725,
        "total_tokens": 1754
      }
    },
    "S4": {
      "cached": false,
      "content": "We don't serve gluten-free bread; would you like a cheese sandwich?",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 16,
        "prompt_tokens": 1167,
        "total_tokens": 1183
      }
    }
  },
  "split": "independent_test"
}
```

## Example 10: independent_test/h014

```json
{
  "assistant_index": 2,
  "author": "claude",
  "categories": {
    "S2": "wrong_arguments",
    "S4": "wrong_arguments"
  },
  "conversation_id": "h014",
  "customer_messages": [
    "A small cappuccino with oat milk and a medium americano with an extra shot"
  ],
  "expected_arguments": [
    {
      "arguments": {
        "items": [
          {
            "item_name": "cappuccino",
            "modifiers": [
              "oat_milk"
            ],
            "quantity": 1,
            "size": "small"
          },
          {
            "item_name": "americano",
            "modifiers": [
              "extra_shot"
            ],
            "quantity": 1,
            "size": "medium"
          }
        ]
      },
      "id": "call_2",
      "name": "create_order"
    }
  ],
  "expected_label": "tool:create_order",
  "intent": "multi_order",
  "responses": {
    "S2": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"Alex Turner\", \"items\": [{\"item_name\": \"cappuccino\", \"quantity\": 1, \"modifiers\": [\"oat_milk\"], \"size\": \"small\"}, {\"item_name\": \"americano\", \"quantity\": 1, \"modifiers\": [\"extra_shot\"], \"size\": \"medium\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 86,
        "prompt_tokens": 1733,
        "total_tokens": 1819
      }
    },
    "S4": {
      "cached": false,
      "content": "<tool_call>\n{\"name\": \"create_order\", \"arguments\": {\"customer_name\": \"John Smith\", \"items\": [{\"item_name\": \"cappuccino\", \"quantity\": 1, \"modifiers\": [\"oat_milk\"], \"size\": \"small\"}, {\"item_name\": \"americano\", \"quantity\": 1, \"modifiers\": [\"extra_shot\"], \"size\": \"medium\"}]}}\n</tool_call>",
      "finish_reason": "stop",
      "tool_calls": [],
      "usage": {
        "completion_tokens": 86,
        "prompt_tokens": 1175,
        "total_tokens": 1261
      }
    }
  },
  "split": "independent_test"
}
```
