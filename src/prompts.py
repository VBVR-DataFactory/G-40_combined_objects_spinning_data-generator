"""
Prompts for Combined Objects Spinning.
"""

import random


PROMPTS = [
    "The scene shows [num] objects on the left side and dashed target outlines on the right side. The dashed target outlines remain completely stationary. For each object, first rotate it in place to match the orientation of its corresponding dashed target outline, then move it horizontally to the right so that it aligns exactly with and fits within its corresponding dashed target outline.",
]


def get_prompt() -> str:
    return random.choice(PROMPTS)


def get_all_prompts() -> list[str]:
    return PROMPTS
