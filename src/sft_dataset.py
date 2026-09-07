"""SFT dataset construction helpers."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


LOG = logging.getLogger(__name__)

STRATEGY_ORDER = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")


# ---------------------------------------------------------------------------
# Strategy prompts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategySpec:
    """Prompt metadata for one controlled execution strategy."""

    strategy_id: str
    name: str
    prompt: str
    output_label: str = "Answer:"


STRATEGY_SPECS: dict[str, StrategySpec] = {
    "S1": StrategySpec(
        strategy_id="S1",
        name="Identifying Correct Information",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S1: Identifying Correct Information.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S1 to resolve any conflict between the reference documents and the model's internal knowledge.\n"
            "Determine which information is the most accurate and reliable, and use it to answer the question.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
    "S2": StrategySpec(
        strategy_id="S2",
        name="Prioritizing External Knowledge",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S2: Prioritizing External Knowledge.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S2.\n"
            "When the reference documents and internal knowledge conflict, prioritize the information contained in the reference documents.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
    "S3": StrategySpec(
        strategy_id="S3",
        name="Generating Corresponding Responses Respectively",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S3: Generating Corresponding Responses Respectively.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S3.\n"
            "Provide two short answers:\n"
            "1. one based on the reference documents;\n"
            "2. one based on the model's internal knowledge.\n\n"
            "Keep the two answers distinct.\n"
            "Output only the two short answers, without explanation.\n\n"
            "Answers:"
        ),
        output_label="Answers:",
    ),
    "S4": StrategySpec(
        strategy_id="S4",
        name="Identifying Correct Information Among Documents",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S4: Identifying Correct Information.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S4 to resolve conflicts among the reference documents.\n"
            "Determine which information is the most accurate and reliable among the conflicting documents, and use it to answer the question.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
    "S5": StrategySpec(
        strategy_id="S5",
        name="Prioritizing High-Frequency Information",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S5: Prioritizing High-Frequency Information.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S5.\n"
            "When the reference documents contain conflicting information, identify which information appears most frequently and prioritize it when answering.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
    "S6": StrategySpec(
        strategy_id="S6",
        name="Generating Corresponding Responses from Conflicting Documents",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S6: Generating Corresponding Responses Respectively.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S6.\n"
            "If the reference documents contain conflicting information, provide a separate short answer corresponding to each conflicting position found in the documents.\n"
            "Output only the short answers, without explanation.\n\n"
            "Answers:"
        ),
        output_label="Answers:",
    ),
    "S7": StrategySpec(
        strategy_id="S7",
        name="Prioritizing External Knowledge with Abstention",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S7: Prioritizing External Knowledge.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S7.\n"
            "Answer the question using the reference documents.\n"
            "If the reference documents do not contain sufficient information to answer the question, output exactly:\n"
            "\"The documents cannot answer this question.\"\n\n"
            "Output only the answer.\n\n"
            "Answer:"
        ),
    ),
    "S8": StrategySpec(
        strategy_id="S8",
        name="External Knowledge with Internal-Knowledge Fallback",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S8: External Knowledge with Internal-Knowledge Fallback.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S8.\n"
            "Use the reference documents to answer the question when they contain relevant information.\n"
            "If the reference documents do not contain the relevant information, use the model's internal knowledge instead.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid JSON object on line {line_number} of {path}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> int:
    """Write records to JSONL and return the number of written lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _stable_bucket(question_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def validate_split_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    total = train_ratio + validation_ratio + test_ratio
    if train_ratio < 0 or validation_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")


def assign_split(question_id: str, seed: int, train_ratio: float, validation_ratio: float, test_ratio: float) -> str:
    bucket = _stable_bucket(question_id, seed)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + validation_ratio:
        return "validation"
    return "test"


def split_question_ids(
    question_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[str]]:
    """Split question ids before any strategy augmentation."""

    validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    split_map = {"train": [], "validation": [], "test": []}
    for question_id in sorted(set(question_ids)):
        split_name = assign_split(question_id, seed, train_ratio, validation_ratio, test_ratio)
        split_map[split_name].append(question_id)
    return split_map


# ---------------------------------------------------------------------------
# Scenario flattening
# ---------------------------------------------------------------------------

def flatten_scenario_documents(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect unique document-like evidence units from one scenario."""

    raw_documents: list[dict[str, Any]] = []
    for conflict in scenario.get("conflicts", []):
        if not isinstance(conflict, dict):
            continue
        for element in conflict.get("conflictual_element", []):
            if isinstance(element, dict):
                raw_documents.append(element)
    for element in scenario.get("non_conflictual_elements", []):
        if isinstance(element, dict):
            raw_documents.append(element)

    seen: set[tuple[Any, ...]] = set()
    documents: list[dict[str, Any]] = []
    for raw in raw_documents:
        key = (
            raw.get("document_id"),
            raw.get("source_type"),
            raw.get("rank"),
            raw.get("retriever"),
            raw.get("text"),
        )
        if key in seen:
            continue
        seen.add(key)
        documents.append(
            {
                "document_id": raw.get("document_id", ""),
                "title": raw.get("title", ""),
                "text": raw.get("text", ""),
                "source_type": raw.get("source_type", ""),
                "rank": raw.get("rank"),
                "score": raw.get("score"),
                "source_corpus": raw.get("source_corpus", ""),
                "retriever": raw.get("retriever", ""),
                "content": raw.get("content"),
            }
        )
    return documents


def assign_document_ids(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, document in enumerate(documents, start=1):
        output.append(
            {
                "id": f"D{index}",
                **document,
            }
        )
    return output


def render_documents(documents: list[dict[str, Any]]) -> str:
    """Render documents in a stable, prompt-friendly format."""

    blocks: list[str] = []
    for document in documents:
        lines = [f"[Document {document['id']}]"]
        title = str(document.get("title", "")).strip()
        if title:
            lines.append(f"Title: {title}")
        text = str(document.get("text", "")).strip()
        lines.append(f"Text: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_execution_prompt(question: str, documents: list[dict[str, Any]], strategy_id: str) -> str:
    """Build the controlled-execution prompt for one strategy."""

    if strategy_id not in STRATEGY_SPECS:
        raise KeyError(f"Unknown strategy_id: {strategy_id}")
    spec = STRATEGY_SPECS[strategy_id]
    rendered_documents = render_documents(documents)
    return spec.prompt.format(documents=rendered_documents, question=question)


def build_execution_messages(question: str, documents: list[dict[str, Any]], strategy_id: str) -> list[dict[str, str]]:
    prompt = build_execution_prompt(question, documents, strategy_id)
    return [
        {
            "role": "system",
            "content": "You are a dataset construction assistant. Follow the assigned strategy exactly.",
        },
        {"role": "user", "content": prompt},
    ]


# ---------------------------------------------------------------------------
# Execution aggregation
# ---------------------------------------------------------------------------

def normalize_execution_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize execution outputs into the aggregator schema."""

    question_id = str(record.get("question_id") or record.get("query_id") or "")
    strategy_id = str(record.get("strategy_id") or "")
    if not question_id:
        raise ValueError("Execution record is missing question_id/query_id.")
    if strategy_id not in STRATEGY_SPECS:
        raise ValueError(f"Execution record has unknown strategy_id: {strategy_id}")

    return {
        "question_id": question_id,
        "strategy_id": strategy_id,
        "answer": record.get("answer", ""),
        "correct": bool(record.get("correct", False)),
        "score": record.get("score"),
        "prompt": record.get("prompt"),
        "scenario": record.get("scenario"),
        "metadata": record.get("metadata", {}),
    }


def aggregate_valid_strategies(execution_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build multi-label question records from per-strategy executions."""

    grouped: dict[str, dict[str, Any]] = {}
    for record in execution_records:
        normalized = normalize_execution_record(record)
        bucket = grouped.setdefault(
            normalized["question_id"],
            {
                "question_id": normalized["question_id"],
                "valid_strategies": [],
                "executions": {},
            },
        )
        bucket["executions"][normalized["strategy_id"]] = {
            "correct": normalized["correct"],
            "answer": normalized["answer"],
            "score": normalized["score"],
        }
        if normalized["correct"]:
            bucket["valid_strategies"].append(normalized["strategy_id"])

    output: list[dict[str, Any]] = []
    for question_id in sorted(grouped):
        bucket = grouped[question_id]
        bucket["valid_strategies"] = [
            strategy_id for strategy_id in STRATEGY_ORDER if strategy_id in set(bucket["valid_strategies"])
        ]
        output.append(bucket)
    return output


# ---------------------------------------------------------------------------
# SFT examples
# ---------------------------------------------------------------------------

def build_execution_candidate(scenario: dict[str, Any], strategy_id: str) -> dict[str, Any]:
    question_id = str(scenario.get("query_id", ""))
    question = str(scenario.get("query", ""))
    documents = assign_document_ids(flatten_scenario_documents(scenario))
    prompt = build_execution_prompt(question, documents, strategy_id)
    return {
        "question_id": question_id,
        "strategy_id": strategy_id,
        "question": question,
        "input": {
            "question": question,
            "documents": documents,
        },
        "documents": documents,
        "prompt": prompt,
        "messages": build_execution_messages(question, documents, strategy_id),
        "scenario": {
            "query_id": question_id,
            "query": question,
            "gold_answer": scenario.get("gold_answer", ""),
            "conflicts": scenario.get("conflicts", []),
            "non_conflictual_elements": scenario.get("non_conflictual_elements", []),
        },
    }


def build_execution_candidates(
    scenarios: list[dict[str, Any]],
    strategy_ids: tuple[str, ...] = STRATEGY_ORDER,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for scenario in scenarios:
        for strategy_id in strategy_ids:
            candidates.append(build_execution_candidate(scenario, strategy_id))
    return candidates


def build_sft_example_from_execution(
    execution_record: dict[str, Any],
    valid_strategies_by_question: dict[str, list[str]],
    include_selection_target: bool = False,
) -> dict[str, Any]:
    normalized = normalize_execution_record(execution_record)
    if not normalized["correct"]:
        raise ValueError("SFT examples must be built from correct execution records.")

    question_id = normalized["question_id"]
    strategy_id = normalized["strategy_id"]
    scenario = normalized.get("scenario") or {}
    question = str(scenario.get("query", execution_record.get("question", "")))
    documents = execution_record.get("documents") or assign_document_ids(
        flatten_scenario_documents(scenario)
    )
    prompt = build_execution_prompt(question, documents, strategy_id)

    example: dict[str, Any] = {
        "question_id": question_id,
        "strategy_id": strategy_id,
        "valid_strategies": valid_strategies_by_question.get(question_id, [strategy_id]),
        "input": {
            "question": question,
            "documents": documents,
        },
        "scenario": scenario,
        "prompt": prompt,
        "output": {
            "answer": normalized["answer"],
        },
        "execution": {
            "answer": normalized["answer"],
            "score": normalized["score"],
            "correct": normalized["correct"],
        },
        "messages": [
            {
                "role": "system",
                "content": "You are a dataset construction assistant. Follow the assigned strategy exactly.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": normalized["answer"]},
        ],
    }
    if include_selection_target:
        example["selection"] = {
            "available_strategies": list(STRATEGY_ORDER),
            "selected_strategy": strategy_id,
            "answer": normalized["answer"],
        }
    return example


def build_valid_strategy_index(label_records: list[dict[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for record in label_records:
        question_id = str(record["question_id"])
        valid_strategies = [str(item) for item in record.get("valid_strategies", [])]
        index[question_id] = valid_strategies
    return index


def build_sft_dataset(
    execution_records: list[dict[str, Any]],
    label_records: list[dict[str, Any]] | None = None,
    include_selection_target: bool = False,
) -> list[dict[str, Any]]:
    """Build SFT-ready examples from validated executions."""

    valid_strategies_by_question = (
        build_valid_strategy_index(label_records) if label_records is not None else {}
    )
    examples: list[dict[str, Any]] = []
    for execution_record in execution_records:
        normalized = normalize_execution_record(execution_record)
        if not normalized["correct"]:
            continue
        examples.append(
            build_sft_example_from_execution(
                execution_record,
                valid_strategies_by_question=valid_strategies_by_question,
                include_selection_target=include_selection_target,
            )
        )
    return examples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT-ready datasets from ARC scenario files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompts = subparsers.add_parser(
        "prompts", help="Expand scenarios into strategy-specific execution prompts."
    )
    prompts.add_argument("scenarios", type=Path, help="Input scenario JSONL file.")
    prompts.add_argument("output", type=Path, help="Output JSONL file or directory.")
    prompts.add_argument(
        "--by-strategy",
        action="store_true",
        help="Write one JSONL file per strategy inside the output directory.",
    )

    labels = subparsers.add_parser(
        "labels", help="Aggregate execution outputs into multi-label strategy sets."
    )
    labels.add_argument("executions", type=Path, help="Input execution JSONL file.")
    labels.add_argument("output", type=Path, help="Output JSONL file.")

    sft = subparsers.add_parser("sft", help="Build SFT splits from execution outputs.")
    sft.add_argument("executions", type=Path, help="Input execution JSONL file.")
    sft.add_argument("output_dir", type=Path, help="Directory where train/validation/test JSONL files are written.")
    sft.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional valid-strategies JSONL file. If omitted, labels are derived from executions.",
    )
    sft.add_argument("--seed", type=int, default=42)
    sft.add_argument("--train-ratio", type=float, default=0.8)
    sft.add_argument("--validation-ratio", type=float, default=0.1)
    sft.add_argument("--test-ratio", type=float, default=0.1)
    sft.add_argument(
        "--selection-target",
        action="store_true",
        help="Include selection-style targets alongside the controlled execution example.",
    )

    return parser.parse_args()


def run_prompt_builder(scenarios_path: Path, output: Path, by_strategy: bool = False) -> None:
    scenarios = load_jsonl(scenarios_path)
    if by_strategy:
        if output.suffix:
            raise ValueError("--by-strategy expects an output directory, not a file path.")
        output.mkdir(parents=True, exist_ok=True)
        for strategy_id in STRATEGY_ORDER:
            strategy_records = [build_execution_candidate(scenario, strategy_id) for scenario in scenarios]
            strategy_path = output / f"{strategy_id.lower()}.jsonl"
            count = write_jsonl(strategy_path, strategy_records)
            LOG.info("Wrote %s prompt records to %s", count, strategy_path)
        return

    records = build_execution_candidates(scenarios)
    count = write_jsonl(output, records)
    LOG.info("Wrote %s prompt records to %s", count, output)


def run_label_builder(executions_path: Path, output: Path) -> None:
    executions = load_jsonl(executions_path)
    labels = aggregate_valid_strategies(executions)
    count = write_jsonl(output, labels)
    LOG.info("Wrote %s label records to %s", count, output)


def run_sft_builder(
    executions_path: Path,
    output_dir: Path,
    labels_path: Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    selection_target: bool = False,
) -> None:
    executions = load_jsonl(executions_path)
    labels = load_jsonl(labels_path) if labels_path is not None else aggregate_valid_strategies(executions)
    valid_strategies_by_question = build_valid_strategy_index(labels)

    correct_executions: list[dict[str, Any]] = []
    for record in executions:
        normalized = normalize_execution_record(record)
        if normalized["correct"]:
            correct_executions.append(normalized)

    question_ids = [record["question_id"] for record in correct_executions]
    split_map = split_question_ids(
        question_ids,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    question_to_split = {
        question_id: split_name
        for split_name, ids in split_map.items()
        for question_id in ids
    }

    split_examples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for execution_record in executions:
        normalized = normalize_execution_record(execution_record)
        if not normalized["correct"]:
            continue
        split_name = question_to_split.get(normalized["question_id"])
        if split_name is None:
            continue
        split_examples[split_name].append(
            build_sft_example_from_execution(
                execution_record,
                valid_strategies_by_question=valid_strategies_by_question,
                include_selection_target=selection_target,
            )
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in split_examples.items():
        split_path = output_dir / f"{split_name}.jsonl"
        count = write_jsonl(split_path, records)
        LOG.info("Wrote %s SFT records to %s", count, split_path)

    labels_path_out = output_dir / "valid_strategies.jsonl"
    labels_count = write_jsonl(labels_path_out, labels)
    LOG.info("Wrote %s label records to %s", labels_count, labels_path_out)


def main() -> None:
    args = parse_args()
    if args.command == "prompts":
        run_prompt_builder(args.scenarios, args.output, by_strategy=args.by_strategy)
    elif args.command == "labels":
        run_label_builder(args.executions, args.output)
    elif args.command == "sft":
        run_sft_builder(
            args.executions,
            args.output_dir,
            labels_path=args.labels,
            seed=args.seed,
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            selection_target=args.selection_target,
        )
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
