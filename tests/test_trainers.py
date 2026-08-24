import pytest

from preference_lab.trainers import PreferenceTrainer, TrainingConfig


@pytest.mark.parametrize("method", ["dpo", "orpo", "mock"])
def test_cpu_trainer_completes_objective_smoke_step(method: str) -> None:
    trainer = PreferenceTrainer(TrainingConfig(method=method))

    trainer.train()

    assert trainer.last_loss is not None
    assert trainer.last_loss >= 0.0


def test_cpu_trainer_rejects_unknown_method() -> None:
    trainer = PreferenceTrainer(TrainingConfig(method="unknown"))

    with pytest.raises(ValueError, match="dpo, orpo, mock"):
        trainer.train()
