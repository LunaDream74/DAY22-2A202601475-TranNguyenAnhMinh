from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import typer
from datasets import Dataset
from peft import LoraConfig
from rich import print
from trl import DPOConfig, DPOTrainer

from preference_lab.data import load_jsonl

app = typer.Typer(help="Fine-tune a causal language model with DPO and LoRA.")


def _load_preferences(path: Path, limit: int | None = None) -> Dataset:
    examples = load_jsonl(path)
    if limit is not None:
        examples = examples[:limit]
    if not examples:
        raise ValueError(f"no valid preference examples found in {path}")
    rows = [
        {
            "prompt": example.prompt,
            "chosen": example.chosen,
            "rejected": example.rejected,
        }
        for example in examples
    ]
    return Dataset.from_list(rows)


def _write_run_config(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.command()
def train(
    model: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    train_file: Path = Path("data/train_preferences.jsonl"),
    validation_file: Path = Path("data/validation_preferences.jsonl"),
    output_dir: Path = Path("outputs/smollm2-135m-dpo-lora"),
    epochs: float = 3.0,
    max_steps: int = -1,
    learning_rate: float = 1e-5,
    beta: float = 0.1,
    max_length: int = 256,
    gradient_accumulation_steps: int = 4,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    target_modules: str = "q_proj,v_proj",
    seed: int = 42,
    smoke: bool = False,
    fp16: bool = False,
    allow_cpu: bool = False,
) -> None:
    """Train and save a LoRA adapter using chosen/rejected preference pairs."""
    use_cuda = torch.cuda.is_available()
    if not use_cuda and not allow_cpu:
        raise RuntimeError("CUDA is unavailable; pass --allow-cpu only if slow CPU training is intended")
    if fp16 and not use_cuda:
        raise ValueError("--fp16 requires CUDA")

    if smoke:
        max_steps = 1
        train_limit = 2
        validation_limit = 2
        gradient_accumulation_steps = 1
    else:
        train_limit = None
        validation_limit = None

    train_dataset = _load_preferences(train_file, train_limit)
    validation_dataset = _load_preferences(validation_file, validation_limit)
    modules = [module.strip() for module in target_modules.split(",") if module.strip()]
    if not modules:
        raise ValueError("target_modules must contain at least one module name")

    dtype = torch.float16 if fp16 else torch.float32
    training_args = DPOConfig(
        output_dir=str(output_dir),
        model_init_kwargs={"dtype": dtype, "low_cpu_mem_usage": True},
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=fp16,
        bf16=False,
        optim="adamw_torch",
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        save_total_limit=2,
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=modules,
        bias="none",
    )

    print(
        f"[cyan]Loading {model} in {dtype} on "
        f"{'GPU: ' + torch.cuda.get_device_name(0) if use_cuda else 'CPU'}[/cyan]"
    )
    print(
        f"[cyan]Training on {len(train_dataset)} pairs; validating on "
        f"{len(validation_dataset)} pairs.[/cyan]"
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(
        output_dir / "run_config.json",
        {
            "model": model,
            "train_file": str(train_file),
            "validation_file": str(validation_file),
            "train_pairs": len(train_dataset),
            "validation_pairs": len(validation_dataset),
            "epochs": epochs,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "beta": beta,
            "max_length": max_length,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": modules,
            "seed": seed,
            "smoke": smoke,
            "fp16": fp16,
            "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
            "dtype": str(dtype),
        },
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    trainer.save_metrics("train", train_result.metrics)

    evaluation_metrics = trainer.evaluate(metric_key_prefix="heldout")
    trainer.save_metrics("heldout", evaluation_metrics)
    print(f"[green]Saved LoRA adapter and metrics to {output_dir}[/green]")


if __name__ == "__main__":
    app()
