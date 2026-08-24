from __future__ import annotations

import os
import tempfile
from pathlib import Path

import typer
from openai import OpenAI
from pydantic import BaseModel, Field
from rich import print

from preference_lab.data import load_jsonl
from preference_lab.schemas import PreferenceExample

app = typer.Typer(help="Synthetic Data Generation for Preference Alignment")

SYSTEM_PROMPT = """You create preference data for DPO and ORPO experiments.
Write accurate, self-contained chosen responses and plausible rejected responses with one
clear quality defect. Keep the rejected response safe: use a subtle factual, reasoning,
relevance, or formatting problem rather than harmful instructions. Vary prompt difficulty
and response length. Prefer applied scenarios, diagnosis, trade-offs, and failure analysis
over introductory definition questions. Do not repeat or paraphrase forbidden topics."""

USER_PROMPT_TEMPLATE = """Generate exactly {count} new preference pairs about {domain}.
Focus on {focus}.

Use these records only as a style guide and do not repeat their prompts:
{examples}

Do not repeat or paraphrase any topic in this list:
{forbidden_prompts}"""


class GeneratedMetadata(BaseModel):
    domain: str = Field(min_length=1)
    rubric: str = Field(min_length=1)


class GeneratedPreference(BaseModel):
    prompt: str = Field(min_length=1)
    chosen: str = Field(min_length=1)
    rejected: str = Field(min_length=1)
    metadata: GeneratedMetadata


class GeneratedPreferenceBatch(BaseModel):
    pairs: list[GeneratedPreference]


def _prompt_key(prompt: str) -> str:
    return " ".join(prompt.split()).casefold()


def _read_api_key(env_file: Path = Path(".env")) -> str | None:
    environment_key = os.getenv("OPENAI_API_KEY")
    if environment_key:
        return environment_key
    if not env_file.exists():
        return None

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", maxsplit=1)
        if name.strip() == "OPENAI_API_KEY":
            api_key = value.strip().strip("\"'")
            return api_key or None
    return None


def _load_style_examples(seed_file: Path, limit: int = 3) -> str:
    if not seed_file.exists():
        return "No seed examples were provided."
    examples = load_jsonl(seed_file)
    return "\n".join(example.model_dump_json() for example in examples[:limit])


def _load_forbidden_prompts(seed_file: Path, output_file: Path) -> str:
    examples = []
    if seed_file.exists():
        examples.extend(load_jsonl(seed_file))
    if output_file.exists():
        examples.extend(load_jsonl(output_file))
    return "\n".join(f"- {example.prompt}" for example in examples)


def _validate_new_pairs(
    generated: GeneratedPreferenceBatch,
    seed_file: Path,
    output_file: Path,
    expected_count: int,
) -> list[PreferenceExample]:
    if len(generated.pairs) != expected_count:
        raise ValueError(
            f"model returned {len(generated.pairs)} pairs; expected exactly {expected_count}"
        )

    existing = load_jsonl(output_file) if output_file.exists() else []
    seeds = load_jsonl(seed_file) if seed_file.exists() else []
    seen_prompts = {_prompt_key(example.prompt) for example in seeds + existing}
    validated: list[PreferenceExample] = []

    for pair in generated.pairs:
        example = PreferenceExample.model_validate(pair.model_dump())
        prompt_key = _prompt_key(example.prompt)
        if prompt_key in seen_prompts:
            raise ValueError(f"generated duplicate prompt: {example.prompt!r}")
        seen_prompts.add(prompt_key)
        validated.append(example)

    return validated


def _append_atomically(output_file: Path, examples: list[PreferenceExample]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing_content = output_file.read_text(encoding="utf-8") if output_file.exists() else ""
    if existing_content and not existing_content.endswith("\n"):
        existing_content += "\n"
    new_content = existing_content + "".join(
        f"{example.model_dump_json()}\n" for example in examples
    )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_file.write(new_content)
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(output_file)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


@app.command()
def generate(
    count: int = 5,
    domain: str = "machine learning",
    focus: str = "technical accuracy and safety",
    output_file: Path = Path("data/synthetic_preferences.jsonl"),
    seed_file: Path = Path("data/sample_preferences.jsonl"),
    model: str = "gpt-4o",
) -> None:
    """Generate and validate synthetic preference pairs using OpenAI."""
    if count <= 0:
        raise typer.BadParameter("count must be greater than zero")

    api_key = _read_api_key()
    if not api_key:
        print("[red]Error: OPENAI_API_KEY is not set in the environment or .env.[/red]")
        raise typer.Exit(1)

    client = OpenAI(api_key=api_key)
    style_examples = _load_style_examples(seed_file)
    forbidden_prompts = _load_forbidden_prompts(seed_file, output_file)
    print(f"Generating [blue]{count}[/blue] pairs for domain: [green]{domain}[/green]...")

    response = client.responses.parse(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=USER_PROMPT_TEMPLATE.format(
            count=count,
            domain=domain,
            focus=focus,
            examples=style_examples,
            forbidden_prompts=forbidden_prompts,
        ),
        text_format=GeneratedPreferenceBatch,
        temperature=0.7,
        store=False,
    )
    generated = response.output_parsed
    if generated is None:
        print("[red]Error: The API returned no parsed preference batch.[/red]")
        raise typer.Exit(1)

    validated = _validate_new_pairs(generated, seed_file, output_file, count)
    _append_atomically(output_file, validated)
    print(f"[green]Successfully added {len(validated)} pairs to {output_file}[/green]")


if __name__ == "__main__":
    app()
