from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .schemas import PreferenceExample


def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return pairwise accuracy, assigning half credit to exact ties."""
    expected = len(examples)
    if len(chosen_scores) != expected or len(rejected_scores) != expected:
        raise ValueError("examples, chosen_scores, and rejected_scores must have identical lengths")
    if not examples:
        return 0.0

    if not all(math.isfinite(score) for score in chosen_scores + rejected_scores):
        raise ValueError("scores must be finite")

    wins = sum(chosen > rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    ties = sum(chosen == rejected for chosen, rejected in zip(chosen_scores, rejected_scores))
    return (wins + 0.5 * ties) / expected


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def unigram_logprob_scores(
    examples: list[PreferenceExample],
) -> tuple[list[float], list[float]]:
    """Score responses with a deterministic add-one-smoothed unigram baseline.

    Mean token log probability avoids making response length the only signal.
    The baseline is intentionally small and is not a substitute for a trained
    preference model, but it provides real, reproducible log-probability scores.
    """
    response_tokens = [
        _TOKEN_PATTERN.findall(response.casefold())
        for example in examples
        for response in (example.chosen, example.rejected)
    ]
    counts = Counter(token for tokens in response_tokens for token in tokens)
    vocabulary_size = len(counts) + 1
    denominator = sum(counts.values()) + vocabulary_size

    def score(tokens: list[str]) -> float:
        if not tokens:
            return math.log(1.0 / denominator)
        return sum(math.log((counts[token] + 1) / denominator) for token in tokens) / len(tokens)

    chosen_scores = [score(tokens) for tokens in response_tokens[::2]]
    rejected_scores = [score(tokens) for tokens in response_tokens[1::2]]
    return chosen_scores, rejected_scores


def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out
