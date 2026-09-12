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
  --output-dir models/sft_arbitration_policy
```

The SFT script does not split the SFT training data. It saves a PEFT/LoRA
adapter that is used as the starting point for GRPO.

## GRPO With Accelerate

```bash
accelerate launch \
  --multi_gpu \
  --num_processes 2 \
  --mixed_precision no \
  scripts/train_grpo_arbitration.py \
  --sft-model-path models/sft_arbitration_policy \
  --train-path data/grpo_train.jsonl \
  --output-dir models/grpo_arbitration_policy \
  --num-generations 4 \
  --per-device-train-batch-size 2
```

For TRL GRPO, the global prompt batch
`num_processes * per_device_train_batch_size` should be divisible by
`num_generations`.

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

The evaluator reports accuracy and macro F1 over A/B/C/D answer letters and
stores completions for inspection.
