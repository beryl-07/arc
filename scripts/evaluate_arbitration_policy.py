#!/usr/bin/env python
"""Evaluate an arbitration policy adapter on ARC scenario JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from src.arbitration_policy import build_arbitration_prompt, extract_gold_letters, extract_response_letters, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/grpo_arbitration_policy"))
    parser.add_argument("--data-path", type=Path, default=Path("data/val.jsonl"))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-document-chars", type=int, default=600)
    parser.add_argument("--max-total-document-chars", type=int, default=3000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.data_path)
    if args.limit > 0:
        records = records[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
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

    predictions: list[str] = []
    references: list[str] = []
    rows: list[dict[str, object]] = []
    for record in tqdm(records, desc=f"Evaluating {args.data_path.name}"):
        prompt = build_arbitration_prompt(
            record,
            max_document_chars=args.max_document_chars,
            max_total_document_chars=args.max_total_document_chars,
        )
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_input_length)
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
        pred = sorted(pred_letters)[0] if pred_letters else "INVALID"
        gold = sorted(gold_letters)[0] if gold_letters else "INVALID"
        predictions.append(pred)
        references.append(gold)
        rows.append(
            {
                "query_id": record.get("query_id"),
                "prediction": pred,
                "gold": gold,
                "correct": bool(pred_letters & gold_letters),
                "completion": completion,
            }
        )

    labels = ["A", "B", "C", "D"]
    accuracy = accuracy_score(references, predictions)
    macro_f1 = f1_score(references, predictions, labels=labels, average="macro", zero_division=0)
    metrics = {"data_path": str(args.data_path), "n": len(records), "accuracy": accuracy, "macro_f1": macro_f1}
    print(json.dumps(metrics, indent=2))

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with args.output_path.open("w", encoding="utf-8") as handle:
            json.dump({"metrics": metrics, "examples": rows}, handle, ensure_ascii=False, indent=2)
        print(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()
