# Arbitration Policy Training Pipeline

This pipeline trains the phase-2 arbitration policy in two stages:

1. SFT/QLoRA on `data/sft_execution_data.jsonl`.
2. GRPO/RLVR from the SFT LoRA adapter on `data/grpo_train.jsonl`.

The held-out files are not recreated from training data. Validation uses
`data/val.jsonl`; final testing uses `data/test.jsonl`.

## Install

```bash
pip install -r requirements.txt
```

## SFT With Accelerate

```bash
accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --mixed_precision no \
  scripts/train_sft_arbitration.py \
  --train-path data/sft_execution_data.jsonl \
  --val-path path/to/sft_val_messages.jsonl \
  --output-dir models/sft_arbitration_policy
```

The SFT script does not split the SFT training data. It saves a PEFT/LoRA
adapter that is used as the starting point for GRPO. `--val-path` is mandatory
and must point to the same conversational `messages` schema as the SFT training
file; the scenario-level `data/val.jsonl` is reserved for held-out arbitration
evaluation.

## GRPO With Accelerate

Single-A100 baseline:

```bash
bash scripts/run_grpo_a100.sh
```

This uses one A100 80GB, 4-bit NF4 QLoRA, BF16, LoRA `r=16`,
`alpha=32`, `dropout=0.05`, target modules `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `num_generations=8`, `max_prompt_length=4096`,
`max_completion_length=256`, `per_device_train_batch_size=1`,
`gradient_accumulation_steps=8`, `learning_rate=1e-6`, `beta=0.01`,
`paged_adamw_8bit`, gradient checkpointing, and no vLLM.

```bash
accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --mixed_precision no \
  scripts/train_grpo_arbitration.py \
  --sft-model-path models/sft_arbitration_policy \
  --train-path data/grpo_train.jsonl \
  --output-dir models/grpo_arbitration_policy
```

The central defaults are now the single-A100 profile. The small-GPU safety
profile is still available for low-VRAM devices and is gated by measured VRAM,
not by a fragile GPU-name check. It does not activate on an A100-class 80GB
device.

Before GRPO starts, prompts are constructed with the model tokenizer and Qwen
chat template, then checked against the 4096-token prompt budget. The system
message, full question/options text, and JSON output instruction are protected.
Only retrieved document text is shortened. If document text compression is not
enough, documents are removed deterministically from the end of the existing
retrieval order. `max_completion_length=256` is never reduced to fix prompt
overflow. If the protected components alone exceed the prompt budget, the
example is rejected with a warning and recorded in
`prompt_budget_diagnostics.json`.

The script also writes `training_config.json` with hardware/software versions,
effective GRPO/LoRA/quantization settings, and prompt-budget statistics.

## Held-Out Evaluation

```bash
python scripts/evaluate_arbitration_policy.py \
  --model-path models/grpo_arbitration_policy \
  --data-path data/val.jsonl \
  --output-path outputs/arbitration_val_predictions.json

python scripts/evaluate_arbitration_policy.py \
  --model-path models/grpo_arbitration_policy \
  --data-path data/test.jsonl \
  --output-path outputs/arbitration_test_predictions.json
```

The evaluator reports subset accuracy, macro F1, macro precision, macro recall,
and the empty prediction rate over multilabel A/B/C/D answer sets. It also
stores completions for inspection.
