"""Utilities for training and evaluating the ARC arbitration policy."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a medical assistant. Output your answer as a valid JSON with "
    "fields: strategy, response, and justification."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def reconstruct_documents(record: dict[str, Any]) -> list[dict[str, str]]:
    """Collect unique retrieved documents from a scenario record."""

    candidates: list[dict[str, Any]] = []
    for conflict in record.get("conflicts", []) or []:
        if not isinstance(conflict, dict):
            continue
        for element in conflict.get("conflictual_element", []) or []:
            if isinstance(element, dict):
                candidates.append(element)
    for element in record.get("non_conflictual_elements", []) or []:
        if isinstance(element, dict):
            candidates.append(element)

    seen: set[str] = set()
    documents: list[dict[str, str]] = []
    for element in candidates:
        if element.get("source_corpus") == "parametric" or element.get("source_type") == "parametric":
            continue
        key = str(element.get("document_id") or element.get("id") or element.get("title") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        title = str(element.get("title") or key)
        text = str(element.get("text") or element.get("content") or "")
        documents.append({"id": key, "title": title, "text": text})
    return documents


def build_arbitration_prompt(
    record: dict[str, Any],
    max_document_chars: int = 600,
    max_total_document_chars: int = 3000,
) -> list[dict[str, str]]:
    """Build the chat prompt used for GRPO and held-out evaluation."""

    documents = record.get("documents")
    if documents is None:
        documents = reconstruct_documents(record)

    parts: list[str] = []
    total = 0
    for index, document in enumerate(documents, start=1):
        text = str(document.get("text", ""))
        clipped = text[:max_document_chars] + ("..." if len(text) > max_document_chars else "")
        title = str(document.get("title") or document.get("id") or f"Document {index}")
        chunk = f"Document {index} [{title}]: {clipped}"
        if total + len(chunk) > max_total_document_chars:
            break
        parts.append(chunk)
        total += len(chunk)

    docs_text = "\n\n".join(parts) if parts else "No retrieved document."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {record['query']}\n\n"
                f"Documents:\n{docs_text}\n\n"
                "Choose the correct answer among A, B, C, D. Output a strict JSON "
                "with strategy, response, and justification."
            ),
        },
    ]


def extract_json(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    cleaned = re.sub(r"```json\s*|```\s*", "", text).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        return " ".join(str(item) for item in value.values())
    return str(value)


def extract_response_letters(completion: Any) -> set[str]:
    """Extract A/B/C/D answer letters from the JSON response field."""

    text = _to_text(completion)
    parsed = extract_json(text)
    if isinstance(parsed, dict):
        response = parsed.get("response", "")
        values = response if isinstance(response, list) else [response]
        letters: set[str] = set()
        for value in values:
            match = re.search(r"\b([A-D])\b", str(value), flags=re.IGNORECASE)
            if match:
                letters.add(match.group(1).upper())
        return letters

    match = re.search(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    return {match.group(1).upper()} if match else set()


def extract_gold_letters(gold_answer: Any) -> set[str]:
    values = gold_answer if isinstance(gold_answer, list) else [gold_answer]
    letters: set[str] = set()
    for value in values:
        match = re.search(r"\b([A-D])\b", str(value), flags=re.IGNORECASE)
        if match:
            letters.add(match.group(1).upper())
    return letters


def exact_letter_reward(completions: list[Any], gold_answer: list[Any], **_: Any) -> list[float]:
    rewards: list[float] = []
    for completion, gold in zip(completions, gold_answer):
        rewards.append(1.0 if extract_response_letters(completion) == extract_gold_letters(gold) else 0.0)
    return rewards


def compute_multilabel_metrics(
    predicted_sets: list[set[str]],
    gold_sets: list[set[str]],
    classes: list[str] | None = None,
) -> dict[str, float]:
    """Calcule les métriques multilabel avec égalité exacte des ensembles."""

    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.preprocessing import MultiLabelBinarizer

    labels = classes or ["A", "B", "C", "D"]
    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform([sorted(items) for items in gold_sets])
    y_pred = mlb.transform([sorted(items) for items in predicted_sets])
    n_examples = len(gold_sets)
    subset_accuracy = (
        sum(predicted == gold for predicted, gold in zip(predicted_sets, gold_sets)) / n_examples
        if n_examples
        else 0.0
    )
    empty_prediction_rate = (
        sum(1 for predicted in predicted_sets if not predicted) / n_examples if n_examples else 0.0
    )
    return {
        "subset_accuracy": float(subset_accuracy),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "empty_prediction_rate": float(empty_prediction_rate),
    }
