"""Human-owned annotation workflow and five clearly excluded worked examples."""

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from cafe_agent.domain.models import BackendState
from cafe_agent.parsing.parser import ParseError, strict_json
from cafe_agent.tools.schemas import validate_arguments

from .builder import ConversationBuilder
from .config import load_data_config
from .io import append_jsonl, read_jsonl
from .schema import Conversation, Intent
from .splitter import normalise_user_text
from .tokenization import load_tokenizer
from .validator import validate_conversation


def load_human_records(path: Path, *, scoreable_only: bool = True) -> list[Conversation]:
    records = [Conversation.model_validate(value) for value in read_jsonl(path)]
    if any(
        record.author not in {"human", "codex_example"} or record.split != "human_test"
        for record in records
    ):
        raise ValueError("Human-test file contains non-human-test records")
    if len({record.id for record in records}) != len(records):
        raise ValueError("Duplicate human-test ID")
    return [record for record in records if record.scoreable] if scoreable_only else records


def seed_examples(path: Path, tokenizer: Any) -> None:
    existing = {record.id for record in load_human_records(path, scoreable_only=False)}
    cases = [
        (Intent.SINGLE_ORDER, "Could I have one medium latte?"),
        (Intent.CLARIFICATION, "I'd like a latte."),
        (Intent.AVAILABILITY, "Do you have orange juice?"),
        (Intent.UNSUPPORTED, "Could you order a pizza?"),
        (Intent.CANCEL_ORDER, "Please order a butter croissant for Casey, then cancel it."),
    ]
    for index, (intent, text) in enumerate(cases, 1):
        record_id = f"codex-example-{index:02d}"
        if record_id in existing:
            continue
        builder = ConversationBuilder()
        try:
            builder.user(text)
            if intent == Intent.SINGLE_ORDER:
                result = builder.call(
                    "create_order",
                    {"items": [{"item_name": "latte", "size": "medium", "quantity": 1}]},
                )
                builder.say(f"Order {result['order_id']} received; total £{result['total']}.")
            elif intent == Intent.CLARIFICATION:
                builder.say("Which size would you like: small, medium or large?", "clarify")
            elif intent == Intent.AVAILABILITY:
                result = builder.call("check_availability", {"item_name": "orange juice"})
                names = ", ".join(item["name"] for item in result["alternatives"])
                builder.say(f"Orange juice is out of stock. Would you like {names}?")
            elif intent == Intent.UNSUPPORTED:
                builder.say("We don't serve pizza. Would you like a cheese sandwich?", "decline")
            else:
                result = builder.call(
                    "create_order",
                    {
                        "customer_name": "Casey",
                        "items": [{"item_name": "butter croissant", "quantity": 1}],
                    },
                )
                builder.call("cancel_order", {"order_id": result["order_id"]})
                builder.say("Your croissant order is cancelled.")
            record = builder.finish(
                record_id=record_id,
                split="human_test",
                intent=intent,
                template_id="human-worked-example",
                author="codex_example",
            )
            validate_conversation(record, tokenizer, load_data_config().max_tokens)
            append_jsonl(path, record.model_dump(mode="json"))
        finally:
            builder.close()


def _annotate_assistant(builder: ConversationBuilder) -> None:
    while True:
        behaviour = input("Expected behaviour [tool/clarify/decline/reply]: ").strip().lower()
        if behaviour == "tool":
            name = input("Tool name: ").strip()
            try:
                arguments = strict_json(input("Arguments (JSON object): "))
                validate_arguments(name, arguments)
            except (ParseError, ValidationError, ValueError, KeyError):
                print("Invalid tool or arguments; try again. No action was executed.")
                continue
            print(json.dumps(builder.call(name, arguments), indent=2))
            print("Now annotate the assistant's next action after this tool result.")
        elif behaviour in {"clarify", "decline", "reply"}:
            text = input("Expected assistant text: ").strip()
            if not text:
                print("Assistant text must not be empty.")
                continue
            builder.say(text, behaviour)
            return
        else:
            print("Choose tool, clarify, decline or reply.")


def annotate(path: Path, tokenizer: Any) -> None:
    config = load_data_config()
    seed_examples(path, tokenizer)
    print("Write your own conversations before seeing model evaluation results.")
    print("Five codex_example records are excluded from scoring. Never paste credentials.")
    print(
        "Commands: /save, /quit, /advance MINUTES. Each conversation starts with a fresh backend."
    )
    while True:
        existing = load_human_records(path, scoreable_only=False)
        count = sum(record.scoreable for record in existing)
        print(
            f"Your scored conversations: {count}; "
            f"target {config.human_min_conversations}–{config.human_max_conversations}."
        )
        if count >= config.human_max_conversations:
            return
        print("Intents: " + ", ".join(intent.value for intent in Intent))
        choice = input("Intent (/quit to exit): ").strip()
        if choice == "/quit":
            return
        try:
            intent = Intent(choice)
        except ValueError:
            print("Choose a listed intent.")
            continue
        builder = ConversationBuilder()
        try:
            while True:
                text = input("Customer message (/save when finished): ").strip()
                if text == "/quit":
                    print("Unsaved conversation discarded.")
                    return
                if text.startswith("/advance "):
                    try:
                        builder.advance(int(text.split(maxsplit=1)[1]))
                    except ValueError:
                        print("Use a non-negative integer minute count.")
                    continue
                if text == "/save":
                    try:
                        record = builder.finish(
                            record_id=f"human-{len(existing) + 1:05d}",
                            split="human_test",
                            intent=intent,
                            template_id="human-authored",
                            author="human",
                        )
                        print("Computed final backend state:")
                        print(record.expected_final_state.model_dump_json(indent=2))
                        expected = input(
                            "Expected final state: Enter to accept, or type JSON: "
                        ).strip()
                        if expected:
                            record.expected_final_state = BackendState.model_validate(
                                strict_json(expected)
                            )
                        validate_conversation(record, tokenizer, config.max_tokens)
                        known = {
                            normalise_user_text(turn)
                            for row in existing
                            for turn in row.scripted_user_turns
                        }
                        keys = [normalise_user_text(turn) for turn in record.scripted_user_turns]
                        if len(keys) != len(set(keys)) or set(keys) & known:
                            raise ValueError(
                                "Duplicate customer text; write a distinct conversation"
                            )
                        append_jsonl(path, record.model_dump(mode="json"))
                    except (ValidationError, ValueError) as exc:
                        print(f"Not saved: {exc}")
                        continue
                    print("Saved and replay-validated.")
                    break
                if not text:
                    continue
                builder.user(text)
                _annotate_assistant(builder)
        finally:
            builder.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=load_data_config().human_path)
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args()
    tokenizer = load_tokenizer()
    if args.seed_only:
        seed_examples(args.path, tokenizer)
    else:
        try:
            annotate(args.path, tokenizer)
        except (EOFError, KeyboardInterrupt):
            print("\nStopped; previously saved conversations remain intact.")


if __name__ == "__main__":
    main()
