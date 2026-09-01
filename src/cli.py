"""Command line entry point for the Phase 1 pipeline."""

from __future__ import annotations

import argparse
import logging

from .config import load_config
from .environment import prepare_runtime, prepare_statpearls
from .logging_utils import configure_logging
from .pipeline import ConflictPipeline


LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 1 conflict scenarios.")
    parser.add_argument(
        "--dataset",
        choices=["pubmedqa", "mmlu-med", "medqa-us", "medmcqa"],
        default="pubmedqa",
    )
    parser.add_argument("--n-questions", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSONL filename.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--mount-drive", action="store_true", help="Mount Google Drive in Colab.")
    parser.add_argument(
        "--prepare-statpearls",
        action="store_true",
        help="Download/extract/chunk StatPearls before retriever initialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    LOG.info("Starting ARC Phase 1 CLI")
    LOG.info("Arguments: dataset=%s n_questions=%s output=%s split=%s", args.dataset, args.n_questions, args.output, args.split)
    config = load_config()
    LOG.info(
        "Config: corpus=%s top_k=%s retrievers=%s output_dir=%s",
        config.resolved_corpus_name,
        config.top_k,
        ",".join(config.retrievers),
        config.output_dir,
    )
    LOG.info("Preparing runtime")
    prepare_runtime(config, mount_drive=args.mount_drive)
    if args.prepare_statpearls:
        LOG.info("Preparing StatPearls before pipeline run")
        prepare_statpearls(config)

    n_questions = args.n_questions or config.default_n_questions
    LOG.info("Initializing pipeline")
    pipeline = ConflictPipeline(config)

    if args.dataset == "pubmedqa":
        output = args.output or f"phase1_scenarios_{n_questions}q.jsonl"
        LOG.info("Running PubMedQA pipeline: split=%s n_questions=%s output=%s", args.split or "train", n_questions, output)
        pipeline.run_pubmedqa(n_questions=n_questions, output_filename=output, split=args.split or "train")
    elif args.dataset == "mmlu-med":
        output = args.output or "mmlu_med_scenarios.jsonl"
        LOG.info("Running MMLU-Med pipeline: split=%s n_questions=%s output=%s", args.split or "test", n_questions, output)
        pipeline.run_mmlu_med(n_questions=n_questions, output_filename=output, split=args.split or "test")
    elif args.dataset == "medqa-us":
        output = args.output or "medqa_us_scenarios.jsonl"
        LOG.info("Running MedQA-US pipeline: split=%s n_questions=%s output=%s", args.split or "test", n_questions, output)
        pipeline.run_medqa_us(n_questions=n_questions, output_filename=output, split=args.split or "test")
    else:  # medmcqa
        output = args.output or "medmcqa_scenarios.jsonl"
        default_split = args.split or "validation"
        LOG.info("Running MedMCQA pipeline: split=%s n_questions=%s output=%s", default_split, n_questions, output)
        pipeline.run_medmcqa(n_questions=n_questions, output_filename=output, split=default_split)
    LOG.info("CLI finished successfully")


if __name__ == "__main__":
    main()
