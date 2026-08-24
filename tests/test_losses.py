import numpy as np
import pytest

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_matches_known_value() -> None:
    loss = dpo_loss(
        np.array([-0.5]),
        np.array([-1.5]),
        np.array([-0.6]),
        np.array([-1.0]),
        beta=0.1,
    )

    assert loss == pytest.approx(float(np.logaddexp(0.0, -0.06)))


def test_dpo_loss_is_stable_for_extreme_margins() -> None:
    loss = dpo_loss(
        np.array([-10_000.0, 0.0]),
        np.array([0.0, -10_000.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        beta=1.0,
    )

    assert np.isfinite(loss)
    assert loss == pytest.approx(5_000.0)


def test_dpo_loss_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        dpo_loss(
            np.array([-0.5, -0.7]),
            np.array([-1.5]),
            np.array([-0.6]),
            np.array([-1.0]),
            beta=0.1,
        )


def test_orpo_loss_matches_known_value() -> None:
    sft_nll = np.array([1.0])
    chosen_logps = np.array([-0.5])
    rejected_logps = np.array([-1.5])
    chosen_log_odds = -0.5 - np.log1p(-np.exp(-0.5))
    rejected_log_odds = -1.5 - np.log1p(-np.exp(-1.5))
    expected = 1.0 + 0.1 * np.logaddexp(0.0, -(chosen_log_odds - rejected_log_odds))

    loss = orpo_loss(sft_nll, chosen_logps, rejected_logps, lambda_orpo=0.1)

    assert loss == pytest.approx(float(expected))


def test_orpo_loss_rejects_positive_log_probability() -> None:
    with pytest.raises(ValueError, match="less than or equal to zero"):
        orpo_loss(np.array([1.0]), np.array([0.1]), np.array([-1.0]), lambda_orpo=0.1)
