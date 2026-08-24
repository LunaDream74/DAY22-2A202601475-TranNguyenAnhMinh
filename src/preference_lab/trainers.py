from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .losses import dpo_loss, orpo_loss


@dataclass(frozen=True)
class TrainingConfig:
    method: str
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2


class PreferenceTrainer:
    """CPU-only objective smoke trainer used by the starter lab.

    It verifies objective wiring on a fixed preference batch. It does not update
    model weights and must be replaced by a TRL-backed trainer for real training.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.last_loss: float | None = None

    def train(self) -> None:
        """Run one deterministic CPU objective smoke step."""
        if self.config.method == "dpo":
            self.last_loss = dpo_loss(
                policy_chosen_logps=np.array([-0.5, -0.8]),
                policy_rejected_logps=np.array([-1.5, -1.1]),
                ref_chosen_logps=np.array([-0.6, -0.7]),
                ref_rejected_logps=np.array([-1.0, -1.0]),
                beta=self.config.beta,
            )
        elif self.config.method == "orpo":
            self.last_loss = orpo_loss(
                sft_nll=np.array([0.5, 0.8]),
                chosen_logps=np.array([-0.5, -0.8]),
                rejected_logps=np.array([-1.5, -1.1]),
                lambda_orpo=self.config.lambda_orpo,
            )
        elif self.config.method == "mock":
            self.last_loss = 0.0
        else:
            raise ValueError("method must be one of: dpo, orpo, mock")
