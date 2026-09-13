"""Utilities for training and evaluating the ARC arbitration policy."""

from __future__ import annotations

import json
import re
import statistics
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SYSTEM_PROMPT = (
    "You are a medical assistant. Output your answer as a valid JSON with "
    "fields: strategy, response, and justification."
)
JSON_INSTRUCTION = (
    "Choose the correct answer among A, B, C, D. Output a strict JSON "
    "with strategy, response, and justification."
)


@dataclass(frozen=True)
class PromptBudgetResult:
    """Auditable result of token-aware arbitration prompt construction."""

    prompt: list[dict[str, str]]
    chat_text: str
    original_token_count: int
    final_token_count: int
    documents_original: int
    documents_final: int
    documents_removed: int
    removed_document_ids: list[str]
    tokens_removed: int
    was_truncated: bool
    was_rejected: bool
    query_id: str | None = None

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("prompt", None)
        data.pop("chat_text", None)
        return data


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


def _normalize_documents(record: dict[str, Any]) -> list[dict[str, str]]:
    documents = record.get("documents")
    if documents is None:
        documents = reconstruct_documents(record)

    normalized: list[dict[str, str]] = []
    for index, document in enumerate(documents or [], start=1):
        if not isinstance(document, dict):
            continue
        key = str(document.get("id") or document.get("document_id") or f"document_{index}")
        title = str(document.get("title") or key)
        text = str(document.get("text") or document.get("content") or "")
        normalized.append({"id": key, "title": title, "text": text})
    return normalized


def _format_document(index: int, document: dict[str, str]) -> str:
    title = str(document.get("title") or document.get("id") or f"Document {index}")
    return f"Document {index} [{title}]: {document.get('text', '')}"


def _build_messages(record: dict[str, Any], documents: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    parts = [_format_document(index, document) for index, document in enumerate(documents, start=1)]
    docs_text = "\n\n".join(parts) if parts else "No retrieved document."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {record['query']}\n\n"
                f"Documents:\n{docs_text}\n\n"
                f"{JSON_INSTRUCTION}"
            ),
        },
    ]


def build_arbitration_prompt(
    record: dict[str, Any],
    max_document_chars: int | None = 600,
    max_total_document_chars: int | None = 3000,
) -> list[dict[str, str]]:
    """Build the chat prompt used for GRPO and held-out evaluation."""

    documents = _normalize_documents(record)

    bounded_documents: list[dict[str, str]] = []
    total = 0
    for index, document in enumerate(documents, start=1):
        text = str(document.get("text", ""))
        if max_document_chars and max_document_chars > 0:
            text = text[:max_document_chars] + ("..." if len(text) > max_document_chars else "")
        candidate = {**document, "text": text}
        chunk = _format_document(index, candidate)
        if max_total_document_chars and max_total_document_chars > 0 and total + len(chunk) > max_total_document_chars:
            break
        bounded_documents.append(candidate)
        total += len(chunk)

    return _build_messages(record, bounded_documents)


def _chat_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _token_count(tokenizer: Any, text: str) -> int:
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def _truncate_text_by_tokens(tokenizer: Any, text: str, token_budget: int) -> str:
    if token_budget <= 0 or not text:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= token_budget:
        return text
    return tokenizer.decode(tokens[:token_budget], skip_special_tokens=True)


def build_token_bounded_arbitration_prompt(
    record: dict[str, Any],
    tokenizer: Any,
    max_prompt_length: int,
    *,
    add_generation_prompt: bool = True,
) -> PromptBudgetResult:
    """Build a tokenizer-verified prompt while protecting non-document fields.

    The system message, question/options text, and final JSON instruction are
    never modified. If needed, retrieved document texts are shortened from their
    ends, then documents are removed deterministically from the end.
    """

    documents = _normalize_documents(record)
    messages = _build_messages(record, documents)
    original_text = _chat_text(tokenizer, messages, add_generation_prompt)
    original_tokens = _token_count(tokenizer, original_text)
    if original_tokens <= max_prompt_length:
        return PromptBudgetResult(
            prompt=messages,
            chat_text=original_text,
            original_token_count=original_tokens,
            final_token_count=original_tokens,
            documents_original=len(documents),
            documents_final=len(documents),
            documents_removed=0,
            removed_document_ids=[],
            tokens_removed=0,
            was_truncated=False,
            was_rejected=False,
            query_id=record.get("query_id"),
        )

    protected_messages = _build_messages(record, [])
    protected_text = _chat_text(tokenizer, protected_messages, add_generation_prompt)
    protected_tokens = _token_count(tokenizer, protected_text)
    if protected_tokens > max_prompt_length:
        warnings.warn(
            "Rejecting example because protected prompt components exceed "
            f"the token budget: query_id={record.get('query_id')} "
            f"protected_tokens={protected_tokens} max_prompt_length={max_prompt_length}",
            RuntimeWarning,
        )
        return PromptBudgetResult(
            prompt=protected_messages,
            chat_text=protected_text,
            original_token_count=original_tokens,
            final_token_count=protected_tokens,
            documents_original=len(documents),
            documents_final=0,
            documents_removed=len(documents),
            removed_document_ids=[document["id"] for document in documents],
            tokens_removed=max(0, original_tokens - protected_tokens),
            was_truncated=True,
            was_rejected=True,
            query_id=record.get("query_id"),
        )

    active = [dict(document) for document in documents]
    removed_ids: list[str] = []

    # Step 1: compress document bodies in token space, preserving order and
    # document headers. Binary search verifies the whole chat-formatted prompt.
    low = 0
    max_doc_tokens = max(
        (_token_count(tokenizer, str(document.get("text", ""))) for document in active),
        default=0,
    )
    best_documents: list[dict[str, str]] | None = None
    best_text = ""
    best_tokens = 0
    while low <= max_doc_tokens:
        mid = (low + max_doc_tokens) // 2
        candidate = [
            {**document, "text": _truncate_text_by_tokens(tokenizer, document.get("text", ""), mid)}
            for document in active
        ]
        candidate_messages = _build_messages(record, candidate)
        candidate_text = _chat_text(tokenizer, candidate_messages, add_generation_prompt)
        candidate_tokens = _token_count(tokenizer, candidate_text)
        if candidate_tokens <= max_prompt_length:
            best_documents = candidate
            best_text = candidate_text
            best_tokens = candidate_tokens
            low = mid + 1
        else:
            max_doc_tokens = mid - 1

    if best_documents is not None:
        return PromptBudgetResult(
            prompt=_build_messages(record, best_documents),
            chat_text=best_text,
            original_token_count=original_tokens,
            final_token_count=best_tokens,
            documents_original=len(documents),
            documents_final=len(best_documents),
            documents_removed=0,
            removed_document_ids=[],
            tokens_removed=max(0, original_tokens - best_tokens),
            was_truncated=True,
            was_rejected=False,
            query_id=record.get("query_id"),
        )

    # Step 2: document headers alone still overflow. Remove documents from the
    # end, preserving earlier retrieved order as the deterministic priority.
    while active:
        removed = active.pop()
        removed_ids.append(removed["id"])
        candidate_messages = _build_messages(record, active)
        candidate_text = _chat_text(tokenizer, candidate_messages, add_generation_prompt)
        candidate_tokens = _token_count(tokenizer, candidate_text)
        if candidate_tokens <= max_prompt_length:
            return PromptBudgetResult(
                prompt=candidate_messages,
                chat_text=candidate_text,
                original_token_count=original_tokens,
                final_token_count=candidate_tokens,
                documents_original=len(documents),
                documents_final=len(active),
                documents_removed=len(removed_ids),
                removed_document_ids=removed_ids,
                tokens_removed=max(0, original_tokens - candidate_tokens),
                was_truncated=True,
                was_rejected=False,
                query_id=record.get("query_id"),
            )

    warnings.warn(
        "Rejecting example after removing all documents because it still exceeds "
        f"the token budget: query_id={record.get('query_id')} "
        f"tokens={protected_tokens} max_prompt_length={max_prompt_length}",
        RuntimeWarning,
    )
    return PromptBudgetResult(
        prompt=protected_messages,
        chat_text=protected_text,
        original_token_count=original_tokens,
        final_token_count=protected_tokens,
        documents_original=len(documents),
        documents_final=0,
        documents_removed=len(documents),
        removed_document_ids=[document["id"] for document in documents],
        tokens_removed=max(0, original_tokens - protected_tokens),
        was_truncated=True,
        was_rejected=True,
        query_id=record.get("query_id"),
    )


def summarize_prompt_budget_results(results: Sequence[PromptBudgetResult], max_prompt_length: int) -> dict[str, Any]:
    """Return reproducible prompt-length and truncation diagnostics."""

    if not results:
        return {
            "max_prompt_length": max_prompt_length,
            "num_examples": 0,
            "num_accepted": 0,
            "num_rejected": 0,
        }

    before = [item.original_token_count for item in results]
    after = [item.final_token_count for item in results if not item.was_rejected]
    tokens_removed = [item.tokens_removed for item in results]
    docs_removed = [item.documents_removed for item in results]

    def pct(values: Sequence[int], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
        return float(ordered[index])

    accepted = [item for item in results if not item.was_rejected]
    return {
        "max_prompt_length": max_prompt_length,
        "num_examples": len(results),
        "num_accepted": len(accepted),
        "num_rejected": len(results) - len(accepted),
        "min_prompt_tokens_before": min(before),
        "mean_prompt_tokens_before": float(statistics.mean(before)),
        "median_prompt_tokens_before": float(statistics.median(before)),
        "p75_prompt_tokens_before": pct(before, 75),
        "p90_prompt_tokens_before": pct(before, 90),
        "p95_prompt_tokens_before": pct(before, 95),
        "p99_prompt_tokens_before": pct(before, 99),
        "max_prompt_tokens_before": max(before),
        "min_prompt_tokens_after": min(after) if after else 0,
        "mean_prompt_tokens_after": float(statistics.mean(after)) if after else 0.0,
        "median_prompt_tokens_after": float(statistics.median(after)) if after else 0.0,
        "p75_prompt_tokens_after": pct(after, 75),
        "p90_prompt_tokens_after": pct(after, 90),
        "p95_prompt_tokens_after": pct(after, 95),
        "p99_prompt_tokens_after": pct(after, 99),
        "max_prompt_tokens_after": max(after) if after else 0,
        "percentage_exceeding_budget_before": 100.0 * sum(1 for value in before if value > max_prompt_length) / len(results),
        "percentage_truncated": 100.0 * sum(1 for item in results if item.was_truncated and not item.was_rejected) / len(results),
        "percentage_rejected": 100.0 * sum(1 for item in results if item.was_rejected) / len(results),
        "average_tokens_removed": float(statistics.mean(tokens_removed)),
        "maximum_tokens_removed": max(tokens_removed),
        "average_documents_removed": float(statistics.mean(docs_removed)),
        "maximum_documents_removed": max(docs_removed),
        "mean_documents_before": float(statistics.mean(item.documents_original for item in results)),
        "mean_documents_after": float(statistics.mean(item.documents_final for item in accepted)) if accepted else 0.0,
    }


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
