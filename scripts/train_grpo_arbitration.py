#!/usr/bin/env python
"""Continue the ARC arbitration policy with GRPO/RLVR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from accelerate import PartialState
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from src.arbitration_policy import build_arbitration_prompt, exact_letter_reward, read_jsonl
from src.config import ARBITRATION_CONFIG


def parse_args() -> argparse.Namespace:
    cfg = ARBITRATION_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-model-path", type=Path, default=cfg.sft_output_dir)
    parser.add_argument("--train-path", type=Path, default=cfg.grpo_train_path)
    parser.add_argument("--output-dir", type=Path, default=cfg.grpo_output_dir)
    parser.add_argument("--max-document-chars", type=int, default=cfg.max_document_chars)
    parser.add_argument("--max-total-document-chars", type=int, default=cfg.max_total_document_chars)
    parser.add_argument("--num-generations", type=int, default=cfg.grpo_num_generations)
    parser.add_argument("--beta", type=float, default=cfg.grpo_beta)
    parser.add_argument("--learning-rate", type=float, default=cfg.grpo_learning_rate)
    parser.add_argument("--max-steps", type=int, default=cfg.grpo_max_steps)
    parser.add_argument("--max-completion-length", type=int, default=cfg.grpo_max_completion_length)
    parser.add_argument("--per-device-train-batch-size", type=int, default=cfg.grpo_per_device_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=cfg.grpo_gradient_accumulation_steps)
    parser.add_argument("--save-steps", type=int, default=cfg.grpo_save_steps)
    parser.add_argument("--logging-steps", type=int, default=cfg.grpo_logging_steps)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    state = PartialState()
    global_batch_size = state.num_processes * args.per_device_train_batch_size
    if global_batch_size % args.num_generations != 0:
        raise ValueError(
            "Invalid GRPO batch configuration: "
            f"num_processes ({state.num_processes}) * per_device_train_batch_size "
            f"({args.per_device_train_batch_size}) = {global_batch_size}, which must be "
            f"divisible by num_generations ({args.num_generations}) for TRL GRPO."
        )

    records = read_jsonl(args.train_path)
    dataset = Dataset.from_list(
        [
            {
                "prompt": build_arbitration_prompt(
                    record,
                    max_document_chars=args.max_document_chars,
                    max_total_document_chars=args.max_total_document_chars,
                ),
                "gold_answer": record["gold_answer"],
            }
            for record in records
        ]
    )
    print(f"GRPO train examples: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.sft_model_path,
        quantization_config=bnb_config,
        device_map={"": state.process_index},
        trust_remote_code=True,
        is_trainable=True,
    )

    # beta active le terme KL natif de TRL ; on utilise les métriques natives
    # journalisées par GRPOTrainer plutôt qu'un second calcul manuel.
    cfg = GRPOConfig(
        output_dir=str(args.output_dir),
        num_generations=args.num_generations,
        beta=args.beta,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        reward_funcs=exact_letter_reward,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"GRPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
