from __future__ import annotations

import tempfile
from pathlib import Path

import typer
from rich import print

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample

app = typer.Typer(help="Prepare leakage-free preference train and validation files.")


def _prompt_key(prompt: str) -> str:
    return " ".join(prompt.split()).casefold()


def _write_jsonl_atomically(path: Path, examples: list[PreferenceExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{example.model_dump_json()}\n" for example in examples)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@app.command()
def prepare(
    original_file: Path = Path("data/sample_preferences.jsonl"),
    synthetic_file: Path = Path("data/synthetic_preferences.jsonl"),
    train_file: Path = Path("data/train_preferences.jsonl"),
    validation_file: Path = Path("data/validation_preferences.jsonl"),
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    """Combine, validate, and split original plus synthetic preferences."""
    examples = load_jsonl(original_file) + load_jsonl(synthetic_file)
    seen_prompts: dict[str, str] = {}
    for example in examples:
        key = _prompt_key(example.prompt)
        if key in seen_prompts:
            raise ValueError(
                f"duplicate prompt across source files: {example.prompt!r} "
                f"matches {seen_prompts[key]!r}"
            )
        seen_prompts[key] = example.prompt

    train, validation = split_by_prompt(
        examples,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    _write_jsonl_atomically(train_file, train)
    _write_jsonl_atomically(validation_file, validation)
    print(
        f"[green]Prepared {len(train)} train and {len(validation)} validation pairs "
        f"from {len(examples)} total.[/green]"
    )


if __name__ == "__main__":
    app()
