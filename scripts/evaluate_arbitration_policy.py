#!/usr/bin/env python
"""Evaluate an arbitration policy adapter on ARC scenario JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from src.arbitration_policy import (
    build_token_bounded_arbitration_prompt,
    compute_multilabel_metrics,
    extract_gold_letters,
    extract_response_letters,
    read_jsonl,
    summarize_prompt_budget_results,
)
from src.config import ARBITRATION_CONFIG


def parse_args() -> argparse.Namespace:
    cfg = ARBITRATION_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=cfg.grpo_output_dir)
    parser.add_argument("--data-path", type=Path, default=cfg.val_path)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-input-length", type=int, default=cfg.eval_max_input_length)
    parser.add_argument("--max-new-tokens", type=int, default=cfg.eval_max_new_tokens)
    parser.add_argument("--max-document-chars", type=int, default=cfg.max_document_chars)
    parser.add_argument("--max-total-document-chars", type=int, default=cfg.max_total_document_chars)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=cfg.grpo_bf16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not ARBITRATION_CONFIG.eval_multilabel:
        raise ValueError("EVAL_MULTILABEL must remain True for arbitration policy evaluation.")

    records = read_jsonl(args.data_path)
    if args.limit > 0:
        records = records[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        is_trainable=False,
    )
    model.eval()

    predicted_sets: list[set[str]] = []
    gold_sets: list[set[str]] = []
    rows: list[dict[str, object]] = []
    prompt_budget_results = []
    for record in tqdm(records, desc=f"Evaluating {args.data_path.name}"):
        budget_result = build_token_bounded_arbitration_prompt(
            record,
            tokenizer=tokenizer,
            max_prompt_length=args.max_input_length,
            add_generation_prompt=True,
        )
        prompt_budget_results.append(budget_result)
        if budget_result.was_rejected:
            continue
        inputs = tokenizer(budget_result.chat_text, return_tensors="pt", truncation=False)
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        pred_letters = extract_response_letters(completion)
        gold_letters = extract_gold_letters(record.get("gold_answer"))
        predicted_sets.append(pred_letters)
        gold_sets.append(gold_letters)
        rows.append(
            {
                "query_id": record.get("query_id"),
                "prediction": sorted(pred_letters),
                "gold": sorted(gold_letters),
                "correct": pred_letters == gold_letters,
                "completion": completion,
                "prompt_token_count": budget_result.final_token_count,
                "prompt_original_token_count": budget_result.original_token_count,
                "documents_removed": budget_result.documents_removed,
            }
        )

    labels = ["A", "B", "C", "D"]
    metrics = {
        "data_path": str(args.data_path),
        "n": len(gold_sets),
        "n_input_records": len(records),
        "classes": labels,
        "prompt_budget": summarize_prompt_budget_results(prompt_budget_results, args.max_input_length),
        **compute_multilabel_metrics(predicted_sets, gold_sets, classes=labels),
    }
    print(json.dumps(metrics, indent=2))

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as handle:
            json.dump({"metrics": metrics, "examples": rows}, handle, ensure_ascii=False, indent=2)
        print(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()
