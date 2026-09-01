"""End-to-end scenario generation pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import PipelineConfig
from .conflicts import (
    detect_conflicts,
    non_conflictual_elements,
    normalize_parametric_elements,
    normalize_retrieved_elements,
)
from .datasets import format_mmlu_mcq, load_mmlu_med, load_pubmedqa
from .nli import NLIClassifier
from .pke import ParametricKnowledgeEstimator
from .retrieval import MedRAGRetrievalManager


LOG = logging.getLogger(__name__)


class ConflictPipeline:
    def __init__(self, config: PipelineConfig, retrieval: MedRAGRetrievalManager | None = None):
        self.config = config
        self.retrieval = retrieval or MedRAGRetrievalManager(config)
        self.nli = NLIClassifier(config)
        self.pke = ParametricKnowledgeEstimator(config)

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

    def run_pubmedqa(
        self, n_questions: int, output_filename: str, split: str = "train"
    ) -> tuple[Path, dict]:
        LOG.info("PubMedQA run started: split=%s n_questions=%s", split, n_questions)
        dataset = load_pubmedqa(split=split)
        output_path = self.config.output_dir / output_filename
        return self._write_scenarios(output_path, self._iter_pubmedqa(dataset, n_questions))

    def run_mmlu_med(
        self, n_questions: int, output_filename: str, split: str = "test"
    ) -> tuple[Path, dict]:
        LOG.info("MMLU-Med run started: split=%s n_questions=%s", split, n_questions)
        dataset = load_mmlu_med(split=split)
        return self._write_scenarios(
            self.config.output_dir / output_filename,
            self._iter_mmlu_med(dataset, n_questions),
        )

    def _iter_pubmedqa(self, dataset, n_questions: int):
        limit = min(n_questions, len(dataset))
        for idx in range(limit):
            row = dataset[idx]
            query_id = str(row["pubid"])
            query = row["question"]
            gold_answer = pubmedqa_gold_answer(row)
            LOG.info("PubMedQA question %s/%s: query_id=%s preview=%r", idx + 1, limit, query_id, query[:100])
            yield self._process_query(query_id, query, gold_answer)

    def _iter_mmlu_med(self, dataset, n_questions: int):
        limit = min(n_questions, len(dataset))
        for idx in range(limit):
            row = dataset[idx]
            query, options, gold_letter = format_mmlu_mcq(row)
            query_id = f"mmlu_med_{row['mmlu_subset']}_{idx}"
            gold_answer = options[gold_letter]
            LOG.info(
                "MMLU-Med question %s/%s: query_id=%s gold=%s preview=%r",
                idx + 1,
                limit,
                query_id,
                gold_letter,
                row["question"][:100],
            )
            yield self._process_query(query_id, query, gold_answer)

    def _write_scenarios(self, output_path: Path, scenarios) -> tuple[Path, dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {"total": 0, "with_conflict": 0, "IC": 0, "CM": 0, "IM": 0}
        LOG.info("Writing scenarios to %s", output_path)

        with output_path.open("w", encoding="utf-8") as handle:
            for scenario in scenarios:
                handle.write(json.dumps(scenario, ensure_ascii=False) + "\n")
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

        LOG.info("Pipeline finished: output_path=%s", output_path)
        LOG.info(
            "Scenario stats: total=%s with_conflict=%s IC=%s CM=%s IM=%s",
            stats["total"],
            stats["with_conflict"],
            stats["IC"],
            stats["CM"],
            stats["IM"],
        )
        return output_path, stats


def pubmedqa_gold_answer(row) -> str:
    if "long_answer" in row and row["long_answer"]:
        return row["long_answer"]
    if "final_decision" in row and row["final_decision"]:
        return row["final_decision"]
    if "answer" in row and row["answer"]:
        return row["answer"]
    return ""
