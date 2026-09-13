#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python scripts/train_grpo_arbitration.py \
  --sft-model-path "${SFT_MODEL_PATH:-models/sft_arbitration_policy}" \
  --train-path "${GRPO_TRAIN_PATH:-data/grpo_train.jsonl}" \
  --output-dir "${GRPO_OUTPUT_DIR:-models/grpo_arbitration_policy_a100}" \
  --num-generations 8 \
  --max-prompt-length 4096 \
  --max-completion-length 256 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-6 \
  --beta 0.01 \
  --optim paged_adamw_8bit \
  --bf16 \
  --no-fp16 \
  --gradient-checkpointing
