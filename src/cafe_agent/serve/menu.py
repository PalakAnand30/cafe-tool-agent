"""Read-only demo menu shortcuts, separate from frozen model evaluation."""

import re
from typing import Any, Literal

from cafe_agent.agent.loop import TraceEntry
from cafe_agent.backend.service import CafeService
from cafe_agent.parsing.parser import ToolCall

MenuGroup = Literal["all", "coffee", "tea", "cold_drinks", "bakery", "sandwiches", "food"]
LABELS = {
    "coffee": "Coffee",
    "tea": "Tea",
    "cold_drinks": "Cold drinks",
    "bakery": "Bakery",
    "sandwiches": "Sandwiches",
}
GROUPS = {**{key: (key,) for key in LABELS}, "all": (None,), "food": ("bakery", "sandwiches")}


def menu_request(message: str) -> MenuGroup | None:
    """Recognize only standalone menu reads; never swallow an order or mixed request."""
    text = " ".join(message.casefold().split()).strip(" .?!")
    pattern = (
        r"(?:please )?(?:(?:show|give) me |can i see |could i see |"
        r"(?:can|could) you show me |what(?:'s| is) on )?(?:the |your )?"
        r"(?:(food|coffee|tea|cold drinks|bakery|sandwiches|sandwich) )?menu(?: please)?"
    )
    match = re.fullmatch(pattern, text)
    if not match:
        return None
    kind = match.group(1) or "all"
    return {"cold drinks": "cold_drinks", "sandwich": "sandwiches"}.get(kind, kind)


def price_label(item: dict[str, Any]) -> str:
    if item["price"] is not None:
        return f"£{item['price']}"
    return " · ".join(f"{size.title()} £{price}" for size, price in item["prices"].items())


def read_menu(cafe: CafeService, group: MenuGroup) -> tuple[list[dict], list[TraceEntry]]:
    items, trace = [], []
    for category in GROUPS[group]:
        arguments = {} if category is None else {"category": category}
        result = cafe.call("get_menu", arguments).model_dump(mode="json")
        trace.append(TraceEntry(call=ToolCall(name="get_menu", arguments=arguments), result=result))
        items.extend(result["items"])
    return items, trace


def menu_reply(items: list[dict], group: MenuGroup) -> str:
    title = "food" if group == "food" else LABELS.get(group, "full").lower()
    sections = [f"Here's our {title} menu, with current prices:"]
    for category, label in LABELS.items():
        members = [item for item in items if item["category"] == category]
        if members:
            sections.append(f"\n**{label}**")
            sections.extend(
                f"- {item['name'].capitalize()} — {price_label(item)}"
                + (" · currently out of stock" if not item["available"] else "")
                for item in members
            )
    sections.append("\nWhat would you like? For drinks, tell me the size too.")
    return "\n".join(sections)
