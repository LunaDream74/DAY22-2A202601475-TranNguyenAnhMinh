from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_batch(name: str, values: np.ndarray) -> NDArray[np.float64]:
    batch = np.asarray(values, dtype=np.float64)
    if batch.ndim != 1 or batch.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(batch)):
        raise ValueError(f"{name} must contain only finite values")
    return batch


def _require_matching_shapes(**batches: np.ndarray) -> None:
    shapes = {batch.shape for batch in batches.values()}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={batch.shape}" for name, batch in batches.items())
        raise ValueError(f"batch shapes must match ({details})")


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """Compute batch DPO loss from sequence log probabilities.

    The objective is ``-log(sigmoid(beta * (policy_ratio - ref_ratio)))``.
    ``logaddexp`` evaluates the equivalent softplus without overflowing.
    """
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and greater than zero")

    policy_chosen = _validate_batch("policy_chosen_logps", policy_chosen_logps)
    policy_rejected = _validate_batch("policy_rejected_logps", policy_rejected_logps)
    ref_chosen = _validate_batch("ref_chosen_logps", ref_chosen_logps)
    ref_rejected = _validate_batch("ref_rejected_logps", ref_rejected_logps)
    _require_matching_shapes(
        policy_chosen=policy_chosen,
        policy_rejected=policy_rejected,
        ref_chosen=ref_chosen,
        ref_rejected=ref_rejected,
    )

    preference_logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    return float(np.mean(np.logaddexp(0.0, -preference_logits)))


def _log_odds(log_probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    if np.any(log_probabilities > 0.0):
        raise ValueError("log probabilities must be less than or equal to zero")

    # A probability of exactly one has infinite odds. Moving it to the nearest
    # representable value below one keeps the finite computation well-defined.
    safe_logps = np.minimum(log_probabilities, -np.finfo(np.float64).eps)
    threshold = -np.log(2.0)
    log_one_minus_p = np.where(
        safe_logps < threshold,
        np.log1p(-np.exp(safe_logps)),
        np.log(-np.expm1(safe_logps)),
    )
    odds: NDArray[np.float64] = safe_logps - log_one_minus_p
    return odds


def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float,
) -> float:
    """Compute a simplified ORPO-style objective.

    The result is mean SFT NLL plus a weighted, stable negative log-sigmoid
    penalty on the chosen-versus-rejected log-odds margin.
    """
    if not np.isfinite(lambda_orpo) or lambda_orpo < 0.0:
        raise ValueError("lambda_orpo must be finite and non-negative")

    sft = _validate_batch("sft_nll", sft_nll)
    chosen = _validate_batch("chosen_logps", chosen_logps)
    rejected = _validate_batch("rejected_logps", rejected_logps)
    _require_matching_shapes(sft_nll=sft, chosen_logps=chosen, rejected_logps=rejected)
    if np.any(sft < 0.0):
        raise ValueError("sft_nll must be non-negative")

    odds_margin = _log_odds(chosen) - _log_odds(rejected)
    preference_penalty = np.logaddexp(0.0, -odds_margin)
    return float(np.mean(sft) + lambda_orpo * np.mean(preference_penalty))
