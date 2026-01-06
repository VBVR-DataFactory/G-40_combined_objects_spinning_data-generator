"""
Prompts for Combined Objects Spinning.
"""

import random


PROMPTS = [
    "Rotate each object in place to match the dashed targets, then slide them horizontally to the right into the targets.",
    "First rotate every shape to the target angle, then move each one to the right into its dashed outline.",
    "Spin the shapes in place until they match the target orientation, then move them right to the dashed positions.",
    "Rotate in place to align with the dashed targets, then translate each object horizontally to the right.",
]


def get_prompt() -> str:
    return random.choice(PROMPTS)


def get_all_prompts() -> list[str]:
    return PROMPTS
