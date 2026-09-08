"""Filter enriched SFT records to keep only questions with valid strategies.

This utility reads ``data/enriched_sft_train.jsonl`` and writes a second JSONL
file containing only the rows whose ``valid_strategies`` field is a non-empty
list.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_jsonl, write_jsonl


LOG = logging.getLogger(__name__)
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "enriched_sft_train.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "dataset_sft_train.jsonl"


def has_valid_strategies(record: dict[str, Any]) -> bool:
    """Return True when the record has at least one valid strategy."""

    valid_strategies = record.get("valid_strategies", [])
    return isinstance(valid_strategies, list) and len(valid_strategies) > 0


def filter_dataset(input_path: Path, output_path: Path) -> dict[str, int]:
    """Filter the enriched dataset and write the resulting subset."""

    records = read_jsonl(input_path)
    kept_records = [record for record in records if has_valid_strategies(record)]

    write_jsonl(output_path, kept_records)

    stats = {
        "total": len(records),
        "kept": len(kept_records),
        "dropped": len(records) - len(kept_records),
    }
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter enriched SFT records into dataset_sft_train.jsonl."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSONL file. Defaults to data/enriched_sft_train.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL file. Defaults to data/dataset_sft_train.jsonl.",
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

    LOG.info("Filtering dataset: input=%s output=%s", args.input_path, args.output_path)
    stats = filter_dataset(args.input_path, args.output_path)

    print("\nSummary")
    print(f"Total rows: {stats['total']}")
    print(f"Kept rows: {stats['kept']}")
    print(f"Dropped rows: {stats['dropped']}")
    LOG.info("Wrote filtered dataset to %s", args.output_path)
    return stats


if __name__ == "__main__":
    main()
