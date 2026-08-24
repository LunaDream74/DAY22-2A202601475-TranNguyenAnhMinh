# Preference Alignment Lab: DPO \& ORPO Starter

Production-style skeleton for a 2-hour lab on preference alignment. The repository is intentionally incomplete: students must implement the logic marked `TODO(student)`.

## Learning goals

- Validate and load preference pairs (`prompt`, `chosen`, `rejected`).
- Implement or wrap DPO/ORPO training logic.
- Build evaluation metrics for pairwise preference and regression prompts.
- Practice production habits: typed code, configs, tests, Makefile, CI, docs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
make test
```

Optional training dependencies:

```bash
pip install -e '.[dev,train]'
```

## Local DPO fine-tuning

Prepare the leakage-free split, run a one-step GPU check, then train the LoRA adapter:

```bash
python scripts/prepare_training_data.py
python scripts/train_dpo.py --smoke --output-dir outputs/smoke-dpo-lora
python scripts/train_dpo.py --output-dir outputs/smollm2-135m-dpo-lora
```

The trainer uses the 51 training pairs and 13 held-out pairs by default. It saves the LoRA
weights, tokenizer, run configuration, checkpoints, and JSON metrics beneath `outputs/`.
Training defaults to FP32 for compatibility with older Pascal GPUs; `--fp16` is available for
hardware where half-precision gradients are stable.

## Lab rules

1. Do not rewrite the whole repository.
2. Implement only the `TODO(student)` blocks unless you have a clear reason.
3. Keep tests passing after each milestone.
4. Do not commit secrets, model weights, or private datasets.

## Milestones

| Time | Goal | Command |
|---|---|---|
| 0-30 min | Setup and inspect sample data | `make test` |
| 30-50 min | Implement dataset validation/collator | `pytest tests/test_data.py` |
| 50-70 min | (Optional) Generate synthetic data | `python scripts/generate_data.py` |
| 70-100 min | Implement DPO or ORPO TODO | `pytest tests/test_losses.py` |
| 100-115 min | Implement evaluation and report | `pref-lab evaluate --config configs/local.yaml` |
| 115-120 min | One-minute demo | `cat outputs/metrics.json` |

## Repository layout

```text
src/preference_lab/     Python package
data/                   Small sample preference dataset
configs/                YAML configs for local experiments
docs/                   Lab guide, rubric, data card template
scripts/                Utility entrypoints
tests/                  Unit tests for student work
```

## Production checklist

- [ ] Dataset schema validated.
- [ ] Train/eval split by prompt, not by row.
- [ ] Config committed; generated artifacts ignored.
- [ ] Metrics saved as JSON.
- [ ] Safety regression prompts run before/after training.
- [ ] Data card updated.
