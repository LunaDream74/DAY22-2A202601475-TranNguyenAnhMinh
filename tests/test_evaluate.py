import pytest

from preference_lab.evaluate import pairwise_accuracy, unigram_logprob_scores
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0


def test_pairwise_accuracy_assigns_half_credit_to_ties() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="a", rejected="b"),
        PreferenceExample(prompt="p2", chosen="c", rejected="d"),
    ]

    assert pairwise_accuracy(examples, [2.0, 1.0], [1.0, 1.0]) == 0.75


def test_pairwise_accuracy_rejects_mismatched_lengths() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]

    with pytest.raises(ValueError, match="identical lengths"):
        pairwise_accuracy(examples, [], [1.0])


def test_unigram_scores_are_finite_and_aligned() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="common tokens", rejected="rare"),
        PreferenceExample(prompt="p2", chosen="common words", rejected="unusual"),
    ]

    chosen_scores, rejected_scores = unigram_logprob_scores(examples)

    assert len(chosen_scores) == len(rejected_scores) == len(examples)
    assert chosen_scores[0] > rejected_scores[0]
