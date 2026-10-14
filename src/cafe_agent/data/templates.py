"""Five wording families per intent; labels and replies never come from an LLM."""

import random
from typing import Any

from cafe_agent.domain.models import MenuItem, load_domain

from .builder import ConversationBuilder
from .config import DataConfig
from .schema import Conversation, Entity, Intent

STYLES = (
    "I'm {name}. Please {clause}.",
    "{name} here; could you {clause}?",
    "For {name}: {clause}, thanks.",
    "Hi, this is {name}; I'd like you to {clause}.",
    "Would you help {name} and {clause}, please?",
)


def template_ids(config: DataConfig, *, test: bool) -> list[str]:
    boundary = config.template_variants - int(
        config.template_variants * config.test_template_fraction
    )
    variants = range(boundary, config.template_variants) if test else range(boundary)
    return [f"{intent}.v{variant}" for intent in Intent for variant in variants]


def generate_example(
    config: DataConfig,
    record_id: str,
    split: str,
    intent: Intent,
    variant: int,
    multi_turn: bool,
    rng: random.Random,
) -> Conversation:
    domain = load_domain()
    name = f"{rng.choice(config.first_names)} {rng.choice(config.last_names)}"
    name_entity = Entity(kind="name", value=name)
    held = {(pair.item_name, pair.modifier) for pair in config.held_out_combinations}
    builder = ConversationBuilder()

    def entity(kind: str, value: str) -> Entity:
        aliases = config.synonyms.get(value, [])
        if kind == "item":
            aliases = [value + "s"]
        return Entity(kind=kind, value=value, aliases=aliases)

    def user(clause: str, facts: list[Entity] | None = None) -> None:
        builder.user(
            STYLES[variant].format(name=name, clause=clause), [name_entity, *(facts or [])]
        )

    def line(item: MenuItem | None = None, *, modifier: bool = False) -> dict[str, Any]:
        item = item or rng.choice([i for i in domain.items if i.available])
        allowed = [m for m in item.modifiers if split == "test" or (item.name, m) not in held]
        selected = [rng.choice(allowed)] if modifier and allowed else []
        result = {
            "item_name": item.name,
            "quantity": rng.choice(config.quantities),
            "modifiers": selected,
        }
        if item.prices:
            result["size"] = rng.choice(list(item.prices)).value
        return result

    def describe(items: list[dict]) -> tuple[str, list[Entity]]:
        phrases, facts = [], []
        for item in items:
            phrase = (
                f"{item['quantity']} " + (item.get("size", "") + " ").lstrip() + item["item_name"]
            )
            facts.extend(
                [entity("quantity", str(item["quantity"])), entity("item", item["item_name"])]
            )
            if item.get("size"):
                facts.append(entity("size", item["size"]))
            if item["modifiers"]:
                phrase += " with " + " and ".join(m.replace("_", " ") for m in item["modifiers"])
                facts.extend(entity("modifier", m) for m in item["modifiers"])
            facts.append(entity("order_line", phrase))
            phrases.append(phrase)
        return " and ".join(phrases), facts

    def place(items: list[dict]) -> dict:
        phrase, facts = describe(items)
        user("order " + phrase, facts)
        result = builder.call("create_order", {"customer_name": name, "items": items})
        if "error" in result:
            alternatives = result["error"]["alternatives"]
            offer = ", ".join(item["name"] for item in alternatives) or "a menu item"
            builder.say(f"{result['error']['message']} Would you like {offer} instead?")
        else:
            builder.say(
                rng.choice(
                    [
                        "Order {order_id} is received: £{total}, ready in {eta_minutes} minutes.",
                        "Thanks! {order_id} totals £{total}; "
                        "preparation takes {eta_minutes} minutes.",
                        "I've placed {order_id}. Total: £{total}. ETA: {eta_minutes} minutes.",
                    ]
                ).format(**result)
            )
        return result

    try:
        if intent in {
            Intent.SINGLE_ORDER,
            Intent.MULTI_ORDER,
            Intent.ORDER_STATUS,
            Intent.CANCEL_ORDER,
        }:
            if intent == Intent.MULTI_ORDER:
                first, second = rng.sample([i for i in domain.items if i.available], 2)
                items = [line(first, modifier=True), line(second, modifier=True)]
                if split == "test" and rng.choice([True, False]):
                    pair = rng.choice(config.held_out_combinations)
                    selected = next(item for item in domain.items if item.name == pair.item_name)
                    items[0] = line(selected)
                    items[0]["modifiers"] = [pair.modifier]
            else:
                items = [line()]
            result = place(items)
            if intent in {Intent.ORDER_STATUS, Intent.CANCEL_ORDER}:
                order_id = result["order_id"]
                if intent == Intent.ORDER_STATUS:
                    builder.advance(
                        rng.choice(
                            [
                                0,
                                domain.preparing_after_minutes,
                                result["eta_minutes"],
                                domain.collected_after_minutes,
                            ]
                        )
                    )
                    user(f"check the status of {order_id}", [entity("order_id", order_id)])
                    result = builder.call("get_order_status", {"order_id": order_id})
                    builder.say(
                        f"Your order is {result['status']}; ETA {result['eta_minutes']} minutes."
                    )
                else:
                    builder.advance(
                        rng.choice([0, domain.preparing_after_minutes, result["eta_minutes"]])
                    )
                    user(f"cancel {order_id}", [entity("order_id", order_id)])
                    result = builder.call("cancel_order", {"order_id": order_id})
                    builder.say(
                        "Your order is cancelled."
                        if "error" not in result
                        else result["error"]["message"]
                    )
        elif intent == Intent.CLARIFICATION:
            item = line(rng.choice([i for i in domain.items if i.available and i.prices]))
            kind = rng.choice(["size", "quantity", "item"])
            incomplete = item.copy()
            if kind == "size":
                incomplete.pop("size")
                phrase, facts = describe([incomplete])
                question = "Which size would you like: small, medium or large?"
            elif kind == "quantity":
                phrase = f"some {item['size']} {item['item_name']}"
                facts = [entity("size", item["size"]), entity("item", item["item_name"])]
                question = "How many would you like?"
            else:
                phrase = f"{item['quantity']} {item['size']} lat drinks"
                facts = [
                    entity("quantity", str(item["quantity"])),
                    entity("size", item["size"]),
                    entity("ambiguous_item", "lat"),
                ]
                question = "Which exact item: latte, iced latte or chai latte?"
                selected = next(i for i in domain.items if i.name == "latte")
                item = line(selected)
            user("order " + phrase, facts)
            builder.say(question, "clarify")
            if multi_turn:
                place([item])
        elif intent == Intent.AVAILABILITY:
            item = rng.choice(domain.items)
            size = rng.choice(list(item.prices)).value if item.prices else None
            args = {"item_name": item.name}
            facts = [entity("item", item.name)]
            if size:
                args["size"] = size
                facts.append(entity("size", size))
            user(f"check availability of {size + ' ' if size else ''}{item.name}", facts)
            result = builder.call("check_availability", args)
            alternatives = ", ".join(i["name"] for i in result["alternatives"])
            builder.say(
                f"{item.name} is available."
                if result["available"]
                else f"{item.name} is out of stock. Would you like {alternatives}?"
            )
        elif intent == Intent.MENU:
            # Sort categories before random sampling so PYTHONHASHSEED cannot alter the dataset.
            categories = sorted({i.category.value for i in domain.items})
            category = rng.choice([None, *categories])
            user(
                "show the " + (category.replace("_", " ") + " menu" if category else "whole menu"),
                [] if category is None else [entity("category", category.replace("_", " "))],
            )
            result = builder.call("get_menu", {} if category is None else {"category": category})
            builder.say("The menu includes " + ", ".join(i["name"] for i in result["items"]) + ".")
        elif intent == Intent.UNSUPPORTED:
            if rng.choice([True, False]):
                item_name = rng.choice(["pizza", "sushi", "hot dog"])
                user(
                    f"order 1 {item_name}", [entity("quantity", "1"), entity("off_menu", item_name)]
                )
                builder.say(
                    f"We don't serve {item_name}; would you like a cheese sandwich?", "decline"
                )
            else:
                user(
                    "order 1 small latte with warmed",
                    [
                        entity("quantity", "1"),
                        entity("size", "small"),
                        entity("item", "latte"),
                        entity("modifier", "warmed"),
                    ],
                )
                builder.say(
                    "Latte does not support warmed; would you like a small latte without it?",
                    "decline",
                )
        elif intent == Intent.TOOL_ERROR:
            if rng.choice([True, False]):
                place([line(rng.choice([item for item in domain.items if not item.available]))])
            else:
                order_id = f"ORD-{rng.randint(10000, 99999):05d}"
                user(f"check the status of {order_id}", [entity("order_id", order_id)])
                result = builder.call("get_order_status", {"order_id": order_id})
                builder.say(
                    result["error"]["message"] + " Please check the ID, or I can show the menu."
                )
        else:
            topic = rng.choice(
                [
                    "tell me the weather",
                    "give me a refund",
                    "handle my complaint",
                    "tell me a joke",
                    "book a flight",
                    "write a poem",
                ]
            )
            user(topic, [entity("unsupported_request", topic)])
            builder.say(
                rng.choice(
                    [
                        "I can only help with café orders. Would you like to see the menu?",
                        "Sorry, I can't help with that. I can help you choose a drink or sandwich.",
                        "Please contact café staff for that. Would you like help with the menu?",
                    ]
                ),
                "decline",
            )
        return builder.finish(
            record_id=record_id, split=split, intent=intent, template_id=f"{intent}.v{variant}"
        )
    finally:
        builder.close()
