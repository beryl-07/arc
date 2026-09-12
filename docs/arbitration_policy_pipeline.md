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

For TRL GRPO, the global prompt batch
`num_processes * per_device_train_batch_size` should be divisible by
`num_generations`. The default configuration uses `num_generations=8`,
`per_device_train_batch_size=1`, and `gradient_accumulation_steps=8`; launch on
8 processes or override the central configuration consciously for smaller
hardware.

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
