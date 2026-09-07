"""Add gold-answer letters to merged scenario records.

This script builds question-to-letter lookups from the existing dataset loaders
in :mod:`src.datasets`, then enriches ``scenarios_merged.jsonl`` so that each
record stores ``gold_answer`` as a list of letters such as ``["B"]``.
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.datasets import (
    format_medmcqa_mcq,
    format_medqa_us_mcq,
    format_mmlu_mcq,
    load_medmcqa,
    load_medqa_us,
    load_mmlu_med,
)
from src.utils import read_jsonl, write_jsonl


LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "scenarios_merged.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "scenarios_merged_with_letters.jsonl"
DEFAULT_MEDMCQA_SPLIT = "validation"
LETTER_ORDER = ("A", "B", "C", "D")


def normalize_text(text: Any) -> str:
    """Normalize text for stable matching."""

    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def question_key(question: str) -> str:
    return normalize_text(question)


def index_to_letter(index: Any) -> str | None:
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(LETTER_ORDER):
        return LETTER_ORDER[idx]
    return None


def ordered_letters(letters: set[str]) -> list[str]:
    return [letter for letter in LETTER_ORDER if letter in letters]


def merge_lookup(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for key, values in source.items():
        target.setdefault(key, set()).update(values)


def add_lookup_entry(lookup: dict[str, set[str]], question: str, letter: str | None) -> None:
    if not letter:
        return
    key = question_key(question)
    if not key:
        return
    lookup.setdefault(key, set()).add(letter.upper())


def build_lookup_from_mmlu() -> dict[str, set[str]]:
    """Build question-to-letter mapping from MMLU-Med test split."""

    lookup: dict[str, set[str]] = {}
    dataset = load_mmlu_med("test")
    for row in dataset:
        query_text, _options, gold_letter = format_mmlu_mcq(row)
        add_lookup_entry(lookup, query_text, gold_letter)
    return lookup


def build_lookup_from_medqa() -> dict[str, set[str]]:
    """Build question-to-letter mapping from MedQA-US test split."""

    lookup: dict[str, set[str]] = {}
    dataset = load_medqa_us("test")
    for row in dataset:
        query_text, _options, gold_letter = format_medqa_us_mcq(row)
        add_lookup_entry(lookup, query_text, gold_letter)
    return lookup


def build_lookup_from_medmcqa(split: str = DEFAULT_MEDMCQA_SPLIT) -> dict[str, set[str]]:
    """Build question-to-letter mapping from MedMCQA."""

    lookup: dict[str, set[str]] = {}
    dataset = load_medmcqa(split)
    for row in dataset:
        query_text, _options, gold_letter = format_medmcqa_mcq(row)
        add_lookup_entry(lookup, query_text, gold_letter)
    return lookup


def build_source_lookups(medmcqa_split: str = DEFAULT_MEDMCQA_SPLIT) -> dict[str, dict[str, set[str]]]:
    """Build a lookup table per dataset source."""

    return {
        "mmlu_med": build_lookup_from_mmlu(),
        "medqa_us": build_lookup_from_medqa(),
        "medmcqa": build_lookup_from_medmcqa(medmcqa_split),
    }


def build_combined_lookup(source_lookups: dict[str, dict[str, set[str]]]) -> dict[str, set[str]]:
    """Combine the dataset-specific question lookups into one table."""

    combined: dict[str, set[str]] = {}
    for lookup in source_lookups.values():
        merge_lookup(combined, lookup)
    return combined


def parse_options(query: str) -> dict[str, str]:
    """Parse ``A. ...`` / ``B. ...`` options from a question string."""

    pattern = re.compile(r"(?ms)^\s*([A-D])\.\s*(.*?)(?=^\s*[A-D]\.\s*|\Z)")
    options: dict[str, str] = {}
    for match in pattern.finditer(query):
        letter = match.group(1).upper()
        option_text = re.sub(r"\s+", " ", match.group(2)).strip()
        if option_text:
            options[letter] = option_text
    return options


def fallback_letters_from_question(query: str, gold_text: Any) -> list[str]:
    """Infer the correct letter(s) by comparing gold text against the options."""

    options = parse_options(query)
    if not options:
        return []

    normalized_gold = normalize_text(gold_text)
    if not normalized_gold:
        return []

    normalized_options = {letter: normalize_text(text) for letter, text in options.items()}

    exact_matches = [
        letter
        for letter, option_text in normalized_options.items()
        if option_text == normalized_gold
    ]
    if exact_matches:
        return ordered_letters(set(exact_matches))

    return []


def infer_letters_from_lookup(
    record: dict[str, Any],
    source_lookups: dict[str, dict[str, set[str]]],
    combined_lookup: dict[str, set[str]],
) -> tuple[list[str], bool]:
    """Return the letter list and whether the fallback was required."""

    query = str(record.get("query", ""))
    query_id = str(record.get("query_id", ""))
    key = question_key(query)

    if query_id.startswith("mmlu_med_"):
        letters = source_lookups["mmlu_med"].get(key)
    elif query_id.startswith("medqa_us_"):
        letters = source_lookups["medqa_us"].get(key)
    elif query_id.startswith("medmcqa_"):
        letters = source_lookups["medmcqa"].get(key)
    else:
        letters = None

    if not letters:
        letters = combined_lookup.get(key)
    if letters:
        return ordered_letters(letters), False

    fallback = fallback_letters_from_question(query, record.get("gold_answer", ""))
    return fallback, True


def process_scenarios(
    input_path: Path,
    output_path: Path,
    source_lookups: dict[str, dict[str, set[str]]],
    combined_lookup: dict[str, set[str]],
) -> dict[str, int]:
    """Read scenarios, add letter gold answers, and write the enriched file."""

    records = read_jsonl(input_path)
    enriched_records: list[dict[str, Any]] = []

    stats = {
        "total": 0,
        "correspondences_found": 0,
        "lookup_matches": 0,
        "fallback_successes": 0,
        "fallback_used": 0,
        "failures": 0,
    }

    for record in tqdm(records, desc="Adding gold letters", unit="scenario"):
        stats["total"] += 1
        letters, used_fallback = infer_letters_from_lookup(record, source_lookups, combined_lookup)
        if letters:
            stats["correspondences_found"] += 1
            if used_fallback:
                stats["fallback_used"] += 1
                stats["fallback_successes"] += 1
            else:
                stats["lookup_matches"] += 1
        else:
            stats["failures"] += 1
            if used_fallback:
                stats["fallback_used"] += 1

        enriched = dict(record)
        enriched["gold_answer"] = letters
        enriched_records.append(enriched)

    write_jsonl(output_path, enriched_records)
    return stats


def print_stats(stats: dict[str, int]) -> None:
    print("\nSummary")
    print(f"Total questions: {stats['total']}")
    print(f"Correspondences found: {stats['correspondences_found']}")
    print(f"Matches from dataset lookups: {stats['lookup_matches']}")
    print(f"Fallback used: {stats['fallback_used']}")
    print(f"Fallback successes: {stats['fallback_successes']}")
    print(f"Failures: {stats['failures']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add gold-answer letters to merged scenario records."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSONL file. Defaults to data/scenarios_merged.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL file. Defaults to data/scenarios_merged_with_letters.jsonl.",
    )
    parser.add_argument(
        "--medmcqa-split",
        default=DEFAULT_MEDMCQA_SPLIT,
        help="MedMCQA split to load when building the lookup. Defaults to validation.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, int]:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    LOG.info("Building gold-letter lookup tables")
    source_lookups = build_source_lookups(args.medmcqa_split)
    combined_lookup = build_combined_lookup(source_lookups)
    LOG.info("Source lookup tables built: mmlu_med=%s medqa_us=%s medmcqa=%s",
             len(source_lookups["mmlu_med"]), len(source_lookups["medqa_us"]), len(source_lookups["medmcqa"]))
    LOG.info("Combined lookup table built: %s normalized questions", len(combined_lookup))

    LOG.info("Processing scenarios file: %s", args.input_path)
    stats = process_scenarios(args.input_path, args.output_path, source_lookups, combined_lookup)
    print_stats(stats)
    LOG.info("Wrote enriched scenarios file: %s", args.output_path)
    return stats


if __name__ == "__main__":
    main()
