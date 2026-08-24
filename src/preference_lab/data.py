from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load and validate unique preference examples from a JSONL file."""
    source = Path(path)
    examples: list[PreferenceExample] = []
    prompt_lines: dict[str, int] = {}

    with source.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON - {exc}") from exc

            try:
                example = PreferenceExample.model_validate(payload)
            except ValidationError as exc:
                raise ValueError(f"{source}:{line_no}: invalid schema - {exc}") from exc

            prompt_key = " ".join(example.prompt.split()).casefold()
            if prompt_key in prompt_lines:
                first_line = prompt_lines[prompt_key]
                raise ValueError(
                    f"{source}:{line_no}: duplicate prompt (first seen on line {first_line})"
                )

            prompt_lines[prompt_key] = line_no
            examples.append(example)

    return examples


def split_by_prompt(
    examples: list[PreferenceExample],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Deterministically split complete prompt groups to avoid leakage."""
    if not 0.0 <= validation_ratio <= 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    groups: dict[str, list[PreferenceExample]] = {}
    for example in examples:
        groups.setdefault(example.prompt, []).append(example)

    prompts = list(groups)
    random.Random(seed).shuffle(prompts)

    if validation_ratio == 0.0:
        train_prompt_count = len(prompts)
    elif validation_ratio == 1.0:
        train_prompt_count = 0
    else:
        train_prompt_count = int(len(prompts) * (1.0 - validation_ratio))
        if len(prompts) > 1:
            train_prompt_count = min(max(1, train_prompt_count), len(prompts) - 1)

    train_prompts = prompts[:train_prompt_count]
    validation_prompts = prompts[train_prompt_count:]
    train = [example for prompt in train_prompts for example in groups[prompt]]
    validation = [example for prompt in validation_prompts for example in groups[prompt]]
    return train, validation
