"""Seeded surface noise that preserves ordering facts."""

import random


def add_noise(text: str, rng: random.Random) -> str:
    candidates = [
        text.lower(),
        text.translate(str.maketrans("", "", ".,;?!")),
        text.replace("Please", "Pleaze").replace("please", "pleaze").replace("could", "coud"),
        "lemme get some help: " + text.replace("thanks", "cheers"),
        text.replace("I'm", "I am").replace("I'd", "I would").replace("For ", "Takeout for "),
    ]
    return rng.choice([candidate for candidate in candidates if candidate != text])
