#!/usr/bin/env python
"""Continue the ARC arbitration policy with GRPO/RLVR."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Contournement d'un bug de compatibilité entre PEFT et Transformers : certaines
# versions de PEFT importent EmbeddingParallel alors que cette classe peut être
# absente côté Transformers. Le sharding tensor-parallel n'est pas utilisé ici.
import peft.utils.save_and_load as _peft_save_load

_peft_save_load._maybe_shard_state_dict_for_tp = lambda *args, **kwargs: None

import torch
import transformers
import trl
from accelerate import PartialState
from datasets import Dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from src.arbitration_policy import build_arbitration_prompt, exact_letter_reward, read_jsonl
from src.config import ARBITRATION_CONFIG
from src.training_progress import ProgressTimeCallback


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
    parser.add_argument("--lr-scheduler-type", type=str, default=cfg.grpo_lr_scheduler_type)
    parser.add_argument("--warmup-ratio", type=float, default=cfg.grpo_warmup_ratio)
    parser.add_argument("--max-steps", type=int, default=cfg.grpo_max_steps)
    parser.add_argument("--max-prompt-length", type=int, default=cfg.grpo_max_prompt_length)
    parser.add_argument("--max-completion-length", type=int, default=cfg.grpo_max_completion_length)
    parser.add_argument("--per-device-train-batch-size", type=int, default=cfg.grpo_per_device_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=cfg.grpo_gradient_accumulation_steps)
    parser.add_argument("--save-steps", type=int, default=cfg.grpo_save_steps)
    parser.add_argument("--logging-steps", type=int, default=cfg.grpo_logging_steps)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def _filter_init_kwargs(cls: type, kwargs: dict[str, object]) -> dict[str, object]:
    """Garde uniquement les arguments acceptés par la version installée."""

    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _build_grpo_trainer(
    model,
    tokenizer,
    grpo_args: GRPOConfig,
    train_dataset: Dataset,
) -> GRPOTrainer:
    """Construit GRPOTrainer avec le nom tokenizer/processing_class disponible."""

    kwargs = {
        "model": model,
        "args": grpo_args,
        "train_dataset": train_dataset,
        "reward_funcs": exact_letter_reward,
    }
    parameters = inspect.signature(GRPOTrainer.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    else:
        raise RuntimeError(
            "This TRL version exposes neither processing_class nor tokenizer "
            "in GRPOTrainer; cannot pass the tokenizer safely."
        )
    return GRPOTrainer(**_filter_init_kwargs(GRPOTrainer, kwargs))


def _bool_env(name: str, default: bool = True) -> bool:
    """Lit un booléen d'environnement pour les garde-fous d'exécution."""

    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def _apply_small_gpu_overrides(args: argparse.Namespace, state: PartialState) -> None:
    """Réduit les paramètres GRPO sur GPU 16 Go pour éviter les OOM."""

    if not _bool_env("GRPO_AUTO_MEMORY_SAFE_ON_SMALL_GPU", True):
        return
    if not torch.cuda.is_available():
        return

    device_index = state.local_process_index
    total_gib = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
    if total_gib > 17:
        return

    cfg = ARBITRATION_CONFIG
    before = {
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
    }
    args.max_steps = max(args.max_steps, cfg.grpo_memory_safe_max_steps)
    args.num_generations = min(args.num_generations, max(2, cfg.grpo_memory_safe_num_generations))
    args.max_prompt_length = min(args.max_prompt_length, cfg.grpo_memory_safe_max_prompt_length)
    args.max_completion_length = min(args.max_completion_length, cfg.grpo_memory_safe_max_completion_length)
    args.per_device_train_batch_size = min(args.per_device_train_batch_size, cfg.grpo_per_device_batch_size)

    if state.is_local_main_process:
        after = {
            "num_generations": args.num_generations,
            "max_steps": args.max_steps,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
        }
        print(
            "GRPO small-GPU memory-safe overrides enabled "
            f"({total_gib:.1f} GiB/device): {before} -> {after}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    state = PartialState()
    _apply_small_gpu_overrides(args, state)
    global_batch_size = state.num_processes * args.per_device_train_batch_size
    if state.is_local_main_process:
        print(f"Transformers version: {transformers.__version__}", flush=True)
        print(f"TRL version: {trl.__version__}", flush=True)
        print(f"GRPO num_processes: {state.num_processes}", flush=True)
        print(f"GRPO per-device batch size: {args.per_device_train_batch_size}", flush=True)
        print(f"GRPO global prompt batch size: {global_batch_size}", flush=True)
        print(f"GRPO num_generations: {args.num_generations}", flush=True)
        print(f"GRPO max_prompt_length: {args.max_prompt_length}", flush=True)
        print(f"GRPO max_completion_length: {args.max_completion_length}", flush=True)
        print(f"GRPO lr_scheduler_type: {args.lr_scheduler_type}", flush=True)
        print(f"GRPO warmup_ratio: {args.warmup_ratio}", flush=True)
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
    print(f"GRPO train examples: {len(dataset)}", flush=True)

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
    # Le cache est inutile en entraînement avec gradient checkpointing et augmente
    # fortement la mémoire utilisée sur les GPU 16 Go.
    model.config.use_cache = False

    # beta active le terme KL natif de TRL ; on utilise les métriques natives
    # journalisées par GRPOTrainer plutôt qu'un second calcul manuel.
    grpo_config_kwargs = {
        "output_dir": str(args.output_dir),
        "num_generations": args.num_generations,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "optim": "paged_adamw_8bit",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "disable_tqdm": False,
        "report_to": "none",
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
    }
    cfg = GRPOConfig(**_filter_init_kwargs(GRPOConfig, grpo_config_kwargs))
    trainer = _build_grpo_trainer(
        model=model,
        tokenizer=tokenizer,
        grpo_args=cfg,
        train_dataset=dataset,
    )
    trainer.add_callback(ProgressTimeCallback("GRPO"))
    print(f"GRPO total steps: {args.max_steps}", flush=True)
    print(f"GRPO effective global prompt batch size: {global_batch_size}", flush=True)
    print(f"GRPO gradient accumulation steps: {args.gradient_accumulation_steps}", flush=True)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"GRPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
