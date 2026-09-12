#!/usr/bin/env python
"""Train the ARC arbitration policy with SFT/QLoRA.

Launch with accelerate, for example:
accelerate launch --multi_gpu --num_processes 2 scripts/train_sft_arbitration.py
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
from pathlib import Path

import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from src.arbitration_policy import read_jsonl
from src.config import ARBITRATION_CONFIG


def parse_args() -> argparse.Namespace:
    cfg = ARBITRATION_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=cfg.model_name)
    parser.add_argument("--train-path", type=Path, default=cfg.sft_train_path)
    parser.add_argument("--val-path", type=Path, default=cfg.sft_val_path)
    parser.add_argument("--output-dir", type=Path, default=cfg.sft_output_dir)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=cfg.sft_max_seq_length)
    parser.add_argument("--epochs", type=float, default=cfg.sft_num_epochs)
    parser.add_argument("--learning-rate", type=float, default=cfg.sft_learning_rate)
    parser.add_argument("--per-device-train-batch-size", type=int, default=cfg.sft_per_device_train_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=cfg.sft_gradient_accumulation_steps)
    parser.add_argument("--save-steps", type=int, default=cfg.sft_save_steps)
    parser.add_argument("--logging-steps", type=int, default=cfg.sft_logging_steps)
    parser.add_argument("--lora-r", type=int, default=cfg.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=cfg.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=cfg.lora_dropout)
    parser.add_argument("--target-modules", nargs="+", default=cfg.lora_target_modules)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def _sft_eval_strategy_kwargs(save_steps: int) -> dict[str, object]:
    """Adapte le nom de l'argument TRL selon la version installée."""

    parameters = inspect.signature(SFTConfig.__init__).parameters
    kwargs: dict[str, object] = {
        "eval_steps": save_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError(
            "This TRL/Transformers version does not expose eval_strategy or "
            "evaluation_strategy in SFTConfig; cannot enable mandatory SFT validation."
        )
    return kwargs


def _filter_init_kwargs(cls: type, kwargs: dict[str, object]) -> dict[str, object]:
    """Garde uniquement les arguments acceptés par la version installée."""

    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _sft_length_kwargs(max_length: int) -> dict[str, object]:
    """Adapte le nom de longueur selon la version TRL."""

    parameters = inspect.signature(SFTConfig.__init__).parameters
    if "max_length" in parameters:
        return {"max_length": max_length}
    if "max_seq_length" in parameters:
        return {"max_seq_length": max_length}
    return {}


def _build_sft_trainer(
    model,
    tokenizer,
    training_args: SFTConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    formatting_func,
) -> SFTTrainer:
    """Construit SFTTrainer avec le nom tokenizer/processing_class disponible."""

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "formatting_func": formatting_func,
    }
    parameters = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "This TRL version exposes neither processing_class nor tokenizer "
            "in SFTTrainer; cannot pass the tokenizer safely."
        )
    return SFTTrainer(**_filter_init_kwargs(SFTTrainer, kwargs))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    if args.val_path is None:
        raise ValueError(
            "SFT validation is mandatory. Provide --val-path or set SFT_VAL_PATH "
            "before launching train_sft_arbitration.py."
        )

    records = read_jsonl(args.train_path)
    val_records = read_jsonl(args.val_path)
    if not records or "messages" not in records[0]:
        raise ValueError("SFT train data must contain conversational `messages` records.")
    if not val_records or "messages" not in val_records[0]:
        raise ValueError("SFT validation data must contain conversational `messages` records.")
    train_dataset = Dataset.from_list(records)
    val_dataset = Dataset.from_list(val_records)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    device_index = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map={"": device_index},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules,
        ),
    )

    def format_chat(example_or_batch: dict):
        messages = example_or_batch["messages"]
        if messages and isinstance(messages[0], list):
            return [tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=False) for item in messages]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    sft_config_kwargs = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        **_sft_eval_strategy_kwargs(args.save_steps),
        "report_to": "none",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "paged_adamw_8bit",
        "dataloader_pin_memory": False,
        **_sft_length_kwargs(args.max_length),
        "packing": False,
        "dataset_num_proc": 2,
        "max_grad_norm": 1.0,
        "seed": args.seed,
    }
    training_args = SFTConfig(**_filter_init_kwargs(SFTConfig, sft_config_kwargs))
    trainer = _build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        formatting_func=format_chat,
    )

    examples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / examples_per_step)
    print(f"SFT examples: {len(train_dataset)}")
    print(f"SFT validation examples: {len(val_dataset)}")
    print(f"Approx. steps/epoch/process: {steps_per_epoch}")

    resume_from = args.resume_from
    if resume_from is None and args.output_dir.exists():
        resume_from = get_last_checkpoint(str(args.output_dir))
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"SFT adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
