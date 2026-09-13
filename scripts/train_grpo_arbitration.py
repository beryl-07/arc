#!/usr/bin/env python
"""Continue the ARC arbitration policy with GRPO/RLVR."""

from __future__ import annotations

import argparse
import inspect
import importlib.metadata
import json
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

from src.arbitration_policy import (
    build_token_bounded_arbitration_prompt,
    exact_letter_reward,
    read_jsonl,
    summarize_prompt_budget_results,
)
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
    parser.add_argument("--optim", type=str, default=cfg.grpo_optimizer)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=cfg.grpo_bf16)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=cfg.grpo_fp16)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=cfg.grpo_gradient_checkpointing,
    )
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _log_gpu_memory(label: str, state: PartialState) -> None:
    if not torch.cuda.is_available():
        if state.is_local_main_process:
            print(f"GPU memory [{label}]: CUDA unavailable", flush=True)
        return
    device_index = state.local_process_index
    torch.cuda.synchronize(device_index)
    allocated = torch.cuda.memory_allocated(device_index) / 1024**3
    reserved = torch.cuda.memory_reserved(device_index) / 1024**3
    peak = torch.cuda.max_memory_allocated(device_index) / 1024**3
    if state.is_local_main_process:
        print(
            f"GPU memory [{label}]: allocated={allocated:.2f} GiB "
            f"reserved={reserved:.2f} GiB peak_allocated={peak:.2f} GiB",
            flush=True,
        )


def _apply_small_gpu_overrides(args: argparse.Namespace, state: PartialState) -> None:
    """Réduit les paramètres GRPO sur GPU 16 Go pour éviter les OOM."""

    if not _bool_env("GRPO_AUTO_MEMORY_SAFE_ON_SMALL_GPU", True):
        return
    if not torch.cuda.is_available():
        return

    device_index = state.local_process_index
    total_gib = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
    if total_gib > ARBITRATION_CONFIG.grpo_small_gpu_vram_gib:
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
    args.optim = cfg.grpo_memory_safe_optimizer

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
            f"({total_gib:.1f} GiB/device, threshold={cfg.grpo_small_gpu_vram_gib:.1f}): "
            f"{before} -> {after}",
            flush=True,
        )


def _validate_precision(args: argparse.Namespace) -> torch.dtype:
    if args.bf16 and args.fp16:
        raise ValueError("Invalid precision configuration: bf16 and fp16 cannot both be enabled.")
    if args.bf16:
        if not torch.cuda.is_available():
            raise RuntimeError("BF16 was requested, but CUDA is not available.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 was requested, but this CUDA device does not report BF16 support.")
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    raise ValueError("For the A100 baseline, enable BF16 explicitly and leave FP16 disabled.")


def _validate_grpo_batch(args: argparse.Namespace, state: PartialState) -> None:
    global_batch_size = state.num_processes * args.per_device_train_batch_size
    if args.per_device_train_batch_size == 1:
        return
    if global_batch_size % args.num_generations != 0:
        raise ValueError(
            "Invalid GRPO batch configuration for this TRL setup: "
            f"num_processes ({state.num_processes}) * per_device_train_batch_size "
            f"({args.per_device_train_batch_size}) = {global_batch_size}, which is not "
            f"divisible by num_generations ({args.num_generations}). The supported "
            "single-A100 baseline uses per_device_train_batch_size=1 with "
            "num_generations=8."
        )


def _prepare_grpo_dataset(records: list[dict[str, object]], tokenizer, args: argparse.Namespace, state: PartialState) -> tuple[Dataset, dict[str, object]]:
    results = [
        build_token_bounded_arbitration_prompt(
            record,
            tokenizer=tokenizer,
            max_prompt_length=args.max_prompt_length,
            add_generation_prompt=True,
        )
        for record in records
    ]
    stats = summarize_prompt_budget_results(results, args.max_prompt_length)
    accepted_rows = [
        {
            "prompt": result.prompt,
            "gold_answer": record["gold_answer"],
            "query_id": record.get("query_id"),
            "prompt_token_count": result.final_token_count,
            "prompt_original_token_count": result.original_token_count,
            "documents_removed": result.documents_removed,
        }
        for record, result in zip(records, results)
        if not result.was_rejected
    ]
    if not accepted_rows:
        raise ValueError("All GRPO examples were rejected by prompt budgeting; cannot train.")

    if state.is_local_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = args.output_dir / "prompt_budget_diagnostics.json"
        diagnostics = {
            "summary": stats,
            "examples": [result.metadata() for result in results],
        }
        diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Prompt budget diagnostics:", json.dumps(stats, indent=2), flush=True)
        print(f"Prompt budget diagnostics saved to {diagnostics_path}", flush=True)

    return Dataset.from_list(accepted_rows), stats


def _save_training_config(args: argparse.Namespace, state: PartialState, prompt_stats: dict[str, object]) -> None:
    if not state.is_local_main_process:
        return
    hardware: dict[str, object] = {"cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(state.local_process_index)
        hardware.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_gib": props.total_memory / 1024**3,
                "cuda_version": torch.version.cuda,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    config = {
        "hardware": hardware,
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "peft": _package_version("peft"),
            "bitsandbytes": _package_version("bitsandbytes"),
            "cuda": torch.version.cuda,
        },
        "grpo": {
            "num_generations": args.num_generations,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "beta": args.beta,
            "optimizer": args.optim,
            "bf16": args.bf16,
            "fp16": args.fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "vllm_enabled": False,
            "max_steps": args.max_steps,
            "seed": args.seed,
        },
        "lora": {
            "r": ARBITRATION_CONFIG.lora_r,
            "alpha": ARBITRATION_CONFIG.lora_alpha,
            "dropout": ARBITRATION_CONFIG.lora_dropout,
            "target_modules": ARBITRATION_CONFIG.lora_target_modules,
        },
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16" if args.bf16 else "float16",
            "bnb_4bit_use_double_quant": True,
        },
        "prompt_budget": prompt_stats,
    }
    path = args.output_dir / "training_config.json"
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Effective GRPO training config saved to {path}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    state = PartialState()
    _apply_small_gpu_overrides(args, state)
    compute_dtype = _validate_precision(args)
    global_batch_size = state.num_processes * args.per_device_train_batch_size
    _validate_grpo_batch(args, state)
    if state.is_local_main_process:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(state.local_process_index)
            print(f"GPU name: {props.name}", flush=True)
            print(f"GPU VRAM: {props.total_memory / 1024**3:.1f} GiB", flush=True)
            print(f"CUDA version: {torch.version.cuda}", flush=True)
            print(f"BF16 supported: {torch.cuda.is_bf16_supported()}", flush=True)
        else:
            print("GPU name: CUDA unavailable", flush=True)
        print(f"PyTorch version: {torch.__version__}", flush=True)
        print(f"Transformers version: {transformers.__version__}", flush=True)
        print(f"TRL version: {trl.__version__}", flush=True)
        print(f"PEFT version: {_package_version('peft')}", flush=True)
        print(f"BitsAndBytes version: {_package_version('bitsandbytes')}", flush=True)
        print(f"GRPO num_processes: {state.num_processes}", flush=True)
        print(f"GRPO per-device batch size: {args.per_device_train_batch_size}", flush=True)
        print(f"GRPO global prompt batch size: {global_batch_size}", flush=True)
        print(f"GRPO num_generations: {args.num_generations}", flush=True)
        print(f"GRPO max_prompt_length: {args.max_prompt_length}", flush=True)
        print(f"GRPO max_completion_length: {args.max_completion_length}", flush=True)
        print(f"GRPO gradient_accumulation_steps: {args.gradient_accumulation_steps}", flush=True)
        print(f"GRPO learning_rate: {args.learning_rate}", flush=True)
        print(f"GRPO beta: {args.beta}", flush=True)
        print(f"GRPO optimizer: {args.optim}", flush=True)
        print(f"GRPO bf16: {args.bf16}", flush=True)
        print(f"GRPO fp16: {args.fp16}", flush=True)
        print(f"GRPO gradient_checkpointing: {args.gradient_checkpointing}", flush=True)
        print(f"LoRA r/alpha/dropout: {ARBITRATION_CONFIG.lora_r}/{ARBITRATION_CONFIG.lora_alpha}/{ARBITRATION_CONFIG.lora_dropout}", flush=True)
        print(f"LoRA target modules: {ARBITRATION_CONFIG.lora_target_modules}", flush=True)
        print("Quantization: 4-bit NF4, double quant, compute dtype bf16" if args.bf16 else "Quantization: 4-bit NF4, double quant, compute dtype fp16", flush=True)
        print("vLLM enabled: False", flush=True)
        print(f"GRPO lr_scheduler_type: {args.lr_scheduler_type}", flush=True)
        print(f"GRPO warmup_ratio: {args.warmup_ratio}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = read_jsonl(args.train_path)
    dataset, prompt_stats = _prepare_grpo_dataset(records, tokenizer, args, state)
    print(f"GRPO train examples accepted: {len(dataset)} / {len(records)}", flush=True)

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
    _log_gpu_memory("after model loading", state)
    # Le cache est inutile en entraînement avec gradient checkpointing et augmente
    # fortement la mémoire utilisée sur les GPU 16 Go.
    if args.gradient_checkpointing:
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
        "optim": args.optim,
        "gradient_checkpointing": args.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "disable_tqdm": False,
        "report_to": "none",
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
        "loss_type": "grpo",
        "mask_truncated_completions": True,
        "use_vllm": False,
    }
    cfg = GRPOConfig(**_filter_init_kwargs(GRPOConfig, grpo_config_kwargs))
    _save_training_config(args, state, prompt_stats)
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
    _log_gpu_memory("before training", state)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    _log_gpu_memory("after training", state)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"GRPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
