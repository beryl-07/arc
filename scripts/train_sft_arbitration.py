#!/usr/bin/env python
"""Train the ARC arbitration policy with SFT/QLoRA.

Launch with accelerate, for example:
accelerate launch --multi_gpu --num_processes 2 scripts/train_sft_arbitration.py
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--train-path", type=Path, default=Path("data/sft_execution_data.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/sft_arbitration_policy"))
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    records = read_jsonl(args.train_path)
    train_dataset = Dataset.from_list(records)
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

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_pin_memory=False,
        max_length=args.max_length,
        packing=False,
        dataset_num_proc=2,
        max_grad_norm=1.0,
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        formatting_func=format_chat,
    )

    examples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_dataset) / examples_per_step)
    print(f"SFT examples: {len(train_dataset)}")
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
