from pathlib import Path

import pytest

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) == 24
    assert examples[0].chosen != examples[0].rejected


def test_error_message_includes_line_number(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt":"a","chosen":"b","rejected":"c"}\n{oops\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r":2:"):
        load_jsonl(bad)


def test_duplicate_prompt_is_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"prompt":"Same prompt","chosen":"a","rejected":"b"}\n'
        '{"prompt":" same  PROMPT ","chosen":"c","rejected":"d"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r":2: duplicate prompt.*line 1"):
        load_jsonl(duplicate)


def test_split_has_no_prompt_leakage() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)

    assert len(train) + len(val) == len(examples)
    assert not ({example.prompt for example in train} & {example.prompt for example in val})


def test_split_keeps_repeated_prompt_groups_together() -> None:
    examples = [
        PreferenceExample(prompt="shared", chosen="a", rejected="b"),
        PreferenceExample(prompt="shared", chosen="c", rejected="d"),
        PreferenceExample(prompt="other", chosen="e", rejected="f"),
    ]

    train, val = split_by_prompt(examples, validation_ratio=0.5)

    assert len(train) + len(val) == len(examples)
    assert not ({example.prompt for example in train} & {example.prompt for example in val})
