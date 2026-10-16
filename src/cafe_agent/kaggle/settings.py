"""Compose training settings without changing frozen data-generation inputs."""

from cafe_agent.config import read_config


def train_settings() -> dict:
    return {**read_config("train.yaml"), **read_config("kaggle.yaml")["training"]}
