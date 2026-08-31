"""Command line entry point for the Phase 1 pipeline."""

from __future__ import annotations

import argparse

from .config import load_config
from .environment import prepare_runtime, prepare_statpearls
from .pipeline import ConflictPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1 conflict scenarios.")
    parser.add_argument("--dataset", choices=["pubmedqa", "mmlu-med"], default="pubmedqa")
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSONL filename.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--mount-drive", action="store_true", help="Mount Google Drive in Colab.")
    parser.add_argument(
        "--prepare-statpearls",
        action="store_true",
        help="Download/extract/chunk StatPearls before retriever initialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    prepare_runtime(config, mount_drive=args.mount_drive)
    if args.prepare_statpearls:
        prepare_statpearls(config)

    n_questions = args.n_questions or config.default_n_questions
    pipeline = ConflictPipeline(config)

    if args.dataset == "pubmedqa":
        output = args.output or f"phase1_scenarios_{n_questions}q.jsonl"
        pipeline.run_pubmedqa(n_questions=n_questions, output_filename=output, split=args.split or "train")
    else:
        output = args.output or "mmlu_med_scenarios.jsonl"
        pipeline.run_mmlu_med(n_questions=n_questions, output_filename=output, split=args.split or "test")


if __name__ == "__main__":
    main()
