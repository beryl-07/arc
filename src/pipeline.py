"""End-to-end scenario generation pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Iterator

from .config import PipelineConfig
from .conflicts import (
    detect_conflicts,
    non_conflictual_elements,
    normalize_parametric_elements,
    normalize_retrieved_elements,
)
from .datasets import (
    format_medmcqa_mcq,
    format_medqa_us_mcq,
    format_mmlu_mcq,
    load_medmcqa,
    load_medqa_us,
    load_mmlu_med,
    load_pubmedqa,
)
from .nli import NLIClassifier
from .pke import ParametricKnowledgeEstimator
from .retrieval import MedRAGRetrievalManager


LOG = logging.getLogger(__name__)

# Type alias for raw question tuples yielded by _iter_* methods.
# Using _process_query only on questions that still need to be processed
# means no GPU work is done for already-completed questions during resume.
_RawQuestion = tuple[str, str, str]  # (query_id, query, gold_answer)


# ---------------------------------------------------------------------------
# Resume helper
# ---------------------------------------------------------------------------

def count_completed_queries(output_path: Path) -> int:
    """Return the number of completed scenario lines in *output_path*.

    Returns ``0`` when the file does not exist or is empty.
    This count is used as a direct ``start_index`` into the dataset iterator
    so that the pipeline jumps straight to the next unprocessed question
    without iterating over already-completed ones.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 0
    try:
        count = 0
        with output_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    count += 1
        return count
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Could not count queries in %s: %s", output_path, exc)
        return 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ConflictPipeline:
    def __init__(self, config: PipelineConfig, retrieval: MedRAGRetrievalManager | None = None):
        self.config = config
        self.retrieval = retrieval or MedRAGRetrievalManager(config)
        self.nli = NLIClassifier(config)
        self.pke = ParametricKnowledgeEstimator(config)

    # ------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------

    def _process_query(self, query_id: str, query: str, gold_answer: str) -> dict:
        LOG.info("Processing query started: query_id=%s", query_id)
        LOG.info("Step 1/4 retrieval started: query_id=%s", query_id)
        retrieval_results = self.retrieval.retrieve_all(query, k=self.config.top_k)
        LOG.info("Step 1/4 retrieval finished: query_id=%s", query_id)
        LOG.info("Step 2/4 PKE started: query_id=%s", query_id)
        pke_result = self.pke.estimate(query)
        LOG.info("Step 2/4 PKE finished: query_id=%s probes=%s", query_id, len(pke_result["probes"]))

        LOG.info("Step 3/4 evidence normalization started: query_id=%s", query_id)
        elements = normalize_retrieved_elements(retrieval_results)
        elements.extend(
            normalize_parametric_elements(
                pke_result["probes"],
                model_name=self.config.pke_model,
            )
        )
        LOG.info("Step 3/4 evidence normalization finished: query_id=%s elements=%s", query_id, len(elements))
        LOG.info("Step 4/4 conflict detection started: query_id=%s", query_id)
        conflicts = detect_conflicts(query_id=query_id, elements=elements, nli=self.nli)
        LOG.info("Step 4/4 conflict detection finished: query_id=%s conflicts=%s", query_id, len(conflicts))

        scenario = {
            "query_id": query_id,
            "query": query,
            "gold_answer": gold_answer,
            "conflicts": conflicts,
            "non_conflictual_elements": non_conflictual_elements(elements, conflicts),
        }
        LOG.info(
            "Processing query finished: query_id=%s non_conflictual_elements=%s",
            query_id,
            len(scenario["non_conflictual_elements"]),
        )
        return scenario

    # ------------------------------------------------------------------
    # Public runners
    # ------------------------------------------------------------------

    def run_pubmedqa(
        self, n_questions: int, output_filename: str, split: str = "train"
    ) -> tuple[Path, dict]:
        LOG.info("PubMedQA run started: split=%s n_questions=%s", split, n_questions)
        dataset = load_pubmedqa(split=split)
        output_path = self.config.output_dir / output_filename
        resume_count = count_completed_queries(output_path)
        return self._write_scenarios(
            output_path,
            self._iter_pubmedqa(dataset, n_questions, start_index=resume_count),
            resume_count=resume_count,
        )

    def run_mmlu_med(
        self, n_questions: int, output_filename: str, split: str = "test"
    ) -> tuple[Path, dict]:
        LOG.info("MMLU-Med run started: split=%s n_questions=%s", split, n_questions)
        dataset = load_mmlu_med(split=split)
        output_path = self.config.output_dir / output_filename
        resume_count = count_completed_queries(output_path)
        return self._write_scenarios(
            output_path,
            self._iter_mmlu_med(dataset, n_questions, start_index=resume_count),
            resume_count=resume_count,
        )

    def run_medqa_us(
        self, n_questions: int, output_filename: str, split: str = "test"
    ) -> tuple[Path, dict]:
        LOG.info("MedQA-US run started: split=%s n_questions=%s", split, n_questions)
        dataset = load_medqa_us(split=split)
        output_path = self.config.output_dir / output_filename
        resume_count = count_completed_queries(output_path)
        return self._write_scenarios(
            output_path,
            self._iter_medqa_us(dataset, n_questions, start_index=resume_count),
            resume_count=resume_count,
        )

    def run_medmcqa(
        self, n_questions: int, output_filename: str, split: str = "validation"
    ) -> tuple[Path, dict]:
        LOG.info("MedMCQA run started: split=%s n_questions=%s", split, n_questions)
        dataset = load_medmcqa(split=split)
        output_path = self.config.output_dir / output_filename
        resume_count = count_completed_queries(output_path)
        return self._write_scenarios(
            output_path,
            self._iter_medmcqa(dataset, n_questions, start_index=resume_count),
            resume_count=resume_count,
        )

    # ------------------------------------------------------------------
    # Raw iterators  (yield tuples, no GPU work)
    # ------------------------------------------------------------------

    def _iter_pubmedqa(self, dataset, n_questions: int, start_index: int = 0) -> Iterator[_RawQuestion]:
        limit = min(n_questions, len(dataset))
        if start_index > 0:
            LOG.info("PubMedQA iterator: skipping to index %s (already completed)", start_index)
        for idx in range(start_index, limit):
            row = dataset[idx]
            query_id = str(row["pubid"])
            query = row["question"]
            gold_answer = pubmedqa_gold_answer(row)
            LOG.debug("PubMedQA question %s/%s: query_id=%s", idx + 1, limit, query_id)
            yield query_id, query, gold_answer

    def _iter_mmlu_med(self, dataset, n_questions: int, start_index: int = 0) -> Iterator[_RawQuestion]:
        limit = min(n_questions, len(dataset))
        if start_index > 0:
            LOG.info("MMLU-Med iterator: skipping to index %s (already completed)", start_index)
        for idx in range(start_index, limit):
            row = dataset[idx]
            query, options, gold_letter = format_mmlu_mcq(row)
            query_id = f"mmlu_med_{row['mmlu_subset']}_{idx}"
            gold_answer = options[gold_letter]
            LOG.debug("MMLU-Med question %s/%s: query_id=%s gold=%s", idx + 1, limit, query_id, gold_letter)
            yield query_id, query, gold_answer

    def _iter_medqa_us(self, dataset, n_questions: int, start_index: int = 0) -> Iterator[_RawQuestion]:
        limit = min(n_questions, len(dataset))
        if start_index > 0:
            LOG.info("MedQA-US iterator: skipping to index %s (already completed)", start_index)
        for idx in range(start_index, limit):
            row = dataset[idx]
            query, options, gold_letter = format_medqa_us_mcq(row)
            query_id = f"medqa_us_{idx}"
            gold_answer = options[gold_letter]
            LOG.debug("MedQA-US question %s/%s: query_id=%s gold=%s", idx + 1, limit, query_id, gold_letter)
            yield query_id, query, gold_answer

    def _iter_medmcqa(self, dataset, n_questions: int, start_index: int = 0) -> Iterator[_RawQuestion]:
        limit = min(n_questions, len(dataset))
        if start_index > 0:
            LOG.info("MedMCQA iterator: skipping to index %s (already completed)", start_index)
        for idx in range(start_index, limit):
            row = dataset[idx]
            query, options, gold_letter = format_medmcqa_mcq(row)
            row_id = row.get("id", str(idx))
            query_id = f"medmcqa_{row_id}"
            gold_answer = options[gold_letter]
            LOG.debug(
                "MedMCQA question %s/%s: query_id=%s gold=%s subject=%s",
                idx + 1, limit, query_id, gold_letter, row.get("subject_name", ""),
            )
            yield query_id, query, gold_answer

    # ------------------------------------------------------------------
    # Writer  (handles resume + append)
    # ------------------------------------------------------------------

    def _write_scenarios(
        self,
        output_path: Path,
        raw_questions: Iterator[_RawQuestion],
        resume_count: int = 0,
    ) -> tuple[Path, dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {"total": 0, "with_conflict": 0, "IC": 0, "CM": 0, "IM": 0}

        # --- Resume detection -------------------------------------------------
        if resume_count > 0:
            LOG.info(
                "Resume detected: %s queries already completed — opening %s in append mode",
                resume_count,
                output_path,
            )
            file_mode = "a"
        else:
            LOG.info("Fresh run: opening %s in write mode", output_path)
            file_mode = "w"
        # ----------------------------------------------------------------------

        with output_path.open(file_mode, encoding="utf-8") as handle:
            for query_id, query, gold_answer in raw_questions:
                # The iterator already starts at the correct index;
                # every question yielded here needs to be processed.
                LOG.info(
                    "Processing question: query_id=%s (resumed_after=%s)",
                    query_id,
                    resume_count,
                )
                scenario = self._process_query(query_id, query, gold_answer)
                handle.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                handle.flush()  # ensure line is on disk before next query starts

                stats["total"] += 1
                if scenario["conflicts"]:
                    stats["with_conflict"] += 1
                for conflict in scenario["conflicts"]:
                    stats[conflict["type"]] += 1

                LOG.info(
                    "Scenario written: query_id=%s conflicts=%s total_written=%s",
                    scenario["query_id"],
                    len(scenario["conflicts"]),
                    stats["total"],
                )

        LOG.info("Pipeline finished: output_path=%s resumed_after=%s", output_path, resume_count)
        LOG.info(
            "Scenario stats: total=%s with_conflict=%s IC=%s CM=%s IM=%s",
            stats["total"],
            stats["with_conflict"],
            stats["IC"],
            stats["CM"],
            stats["IM"],
        )
        return output_path, stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pubmedqa_gold_answer(row) -> str:
    if "long_answer" in row and row["long_answer"]:
        return row["long_answer"]
    if "final_decision" in row and row["final_decision"]:
        return row["final_decision"]
    if "answer" in row and row["answer"]:
        return row["answer"]
    return ""
