"""Escaped menu and receipt cards, with original local vector artwork."""

from decimal import Decimal
from html import escape
from pathlib import Path
from typing import Any

from cafe_agent.backend.service import CafeService

from .menu import GROUPS, MenuGroup, read_menu

ASSETS = Path(__file__).with_name("assets")


def artwork(name: str) -> str:
    return (ASSETS / f"{name}.svg").read_text()


def masthead() -> str:
    return """<div class="masthead"><div class="wordmark"><span class="brand-mark">c.</span>
      <div>the café counter<span>COFFEE, CONVERSATION &amp; A LITTLE COMFORT</span></div></div>
      <div class="masthead-note"><span class="green-dot"></span>
      Your neighbourhood café</div></div>"""


def hero() -> str:
    return (
        """<section class="cafe-hero"><div class="hero-copy"><span class="eyebrow">
      A LITTLE PAUSE IN YOUR DAY</span><h1>Good coffee.<br>A little <em>conversation.</em></h1>
      <p>Find your usual. Discover a new favourite.<br>We'll take it from here.</p></div>
      <div class="hero-art">"""
        + artwork("coffee")
        + """</div>
      <span class="hero-note">made for your coffee moment</span></section>"""
    )


def menu_cards(group: MenuGroup | str, featured: list[str]) -> str:
    cafe = CafeService()
    try:
        items, _ = read_menu(cafe, group if group in GROUPS else "all")
    finally:
        cafe.close()
    if group == "featured":
        items = sorted(
            [item for item in items if item["name"] in featured],
            key=lambda item: featured.index(item["name"]),
        )
    cards = []
    for item in items:
        category = item["category"]
        image = {
            "bakery": "bakery",
            "sandwiches": "sandwich",
            "tea": "tea",
            "cold_drinks": "cold",
        }.get(category, "coffee")
        for keyword, asset in [("muffin", "muffin"), ("brownie", "brownie"), ("roll", "roll")]:
            if keyword in item["name"]:
                image = asset
        price = (
            item["price"]
            if item["price"] is not None
            else min(item["prices"].values(), key=Decimal)
        )
        prefix = "" if item["price"] is not None else "from "
        badge = "" if item["available"] else '<span class="sold-out">Out of stock</span>'
        cards.append(
            f'<article class="menu-card {category}"><div class="menu-art">'
            + artwork(image)
            + badge
            + '</div><div class="menu-card-copy">'
            + f"<h3>{escape(item['name'].capitalize())}</h3>"
            + f"<p>{prefix}<strong>£{escape(price)}</strong></p></div></article>"
        )
    return '<div class="menu-gallery" aria-label="Café menu">' + "".join(cards) + "</div>"


def order_board(state: dict[str, Any]) -> str:
    orders = state["orders"]
    if not orders:
        return (
            """<div class="empty-order"><div class="empty-cup">"""
            + artwork("coffee")
            + """</div><h3>Your next little treat.</h3><p>Tell us what you're craving.<br>
                Your orders will appear here.</p><span class="receipt-rule"></span>
                <div class="empty-total"><span>Nothing ordered yet</span><strong>£0.00</strong>
                </div></div>"""
        )
    cards = []
    for order in orders:
        status = escape(order["status"])
        lines = []
        for line in order["items"]:
            details = [str(line["size"]).title()] if line.get("size") else []
            details.extend(value.replace("_", " ") for value in line["modifiers"])
            lines.append(
                '<div class="receipt-item"><span class="quantity">'
                + str(line["quantity"])
                + "</span><div><strong>"
                + escape(line["item_name"].capitalize())
                + "</strong><small>"
                + escape(" · ".join(details))
                + "</small></div><b>£"
                + escape(line["line_total"])
                + "</b></div>"
            )
        cards.append(
            f'<article class="receipt {status}"><header><span>{escape(order["order_id"])}</span>'
            f'<span class="order-status {status}">{status.capitalize()}</span></header>'
            + "".join(lines)
            + "<footer><span>Order total</span><strong>£"
            + escape(order["total"])
            + "</strong></footer></article>"
        )
    return '<div class="order-receipts">' + "".join(cards) + "</div>"
