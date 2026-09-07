"""Summarize mutually exclusive question sets from a scenario JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize mutually exclusive conflict sets from an ARC JSONL dataset."
    )
    parser.add_argument("dataset_path", type=Path, help="Path to the JSONL dataset.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the summary as JSON instead of a human-readable table.",
    )
    return parser.parse_args()


def summarize_conflicts(dataset_path: Path) -> dict[str, Any]:
    exclusive_sets: Counter[str] = Counter()
    questions_by_type: Counter[str] = Counter()
    total_questions = 0

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            total_questions += 1
            conflicts = record.get("conflicts", [])
            if not isinstance(conflicts, list):
                raise ValueError(
                    f"Invalid 'conflicts' field on line {line_number}: expected a list."
                )

            question_types = set()
            for conflict in conflicts:
                if not isinstance(conflict, dict):
                    raise ValueError(
                        f"Invalid conflict on line {line_number}: expected an object."
                    )
                question_types.add(str(conflict.get("type", "UNKNOWN")))

            set_name = exclusive_set_name(question_types)
            exclusive_sets[set_name] += 1
            for conflict_type in question_types:
                questions_by_type[conflict_type] += 1

    return {
        "dataset_path": str(dataset_path),
        "unit": "question",
        "total_questions": total_questions,
        "questions_with_conflict": total_questions - exclusive_sets["NO_CONFLICT"],
        "questions_without_conflict": exclusive_sets["NO_CONFLICT"],
        "exclusive_question_sets": dict(sorted(exclusive_sets.items())),
        "questions_by_type_non_exclusive": dict(sorted(questions_by_type.items())),
    }


def exclusive_set_name(conflict_types: set[str]) -> str:
    if not conflict_types:
        return "NO_CONFLICT"
    return "_".join(sorted(conflict_types))


def print_table(summary: dict[str, Any]) -> None:
    print(f"Dataset: {summary['dataset_path']}")
    print(f"Unit: {summary['unit']}")
    print(f"Questions: {summary['total_questions']}")
    print(f"Questions with conflict: {summary['questions_with_conflict']}")
    print(f"Questions without conflict: {summary['questions_without_conflict']}")
    print()
    print("Exclusive question sets")
    print("-----------------------")

    exclusive_sets = summary["exclusive_question_sets"]
    width = max(len(set_name) for set_name in exclusive_sets) if exclusive_sets else 0
    for set_name, count in exclusive_sets.items():
        print(f"{set_name:<{width}}  {count}")

    print()
    print("Questions by type (non-exclusive)")
    print("---------------------------------")
    questions_by_type = summary["questions_by_type_non_exclusive"]
    if not questions_by_type:
        print("No conflict types found.")
        return
    width = max(len(conflict_type) for conflict_type in questions_by_type)
    for conflict_type, count in questions_by_type.items():
        print(f"{conflict_type:<{width}}  {count}")


def main() -> None:
    args = parse_args()
    summary = summarize_conflicts(args.dataset_path)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_table(summary)


if __name__ == "__main__":
    main()
