"""End-to-end scenario generation pipelines."""

from __future__ import annotations

import json
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


class ConflictPipeline:
    def __init__(self, config: PipelineConfig, retrieval: MedRAGRetrievalManager | None = None):
        self.config = config
        self.retrieval = retrieval or MedRAGRetrievalManager(config)
        self.nli = NLIClassifier(config)
        self.pke = ParametricKnowledgeEstimator(config)

    def _process_query(self, query_id: str, query: str, gold_answer: str) -> dict:
        retrieval_results = self.retrieval.retrieve_all(query, k=self.config.top_k)
        pke_result = self.pke.estimate(query)

        elements = normalize_retrieved_elements(retrieval_results)
        elements.extend(
            normalize_parametric_elements(
                pke_result["probes"],
                model_name=self.config.pke_model,
            )
        )
        conflicts = detect_conflicts(query_id=query_id, elements=elements, nli=self.nli)

        return {
            "query_id": query_id,
            "query": query,
            "gold_answer": gold_answer,
            "conflicts": conflicts,
            "non_conflictual_elements": non_conflictual_elements(elements, conflicts),
        }

    def run_pubmedqa(
        self, n_questions: int, output_filename: str, split: str = "train"
    ) -> tuple[Path, dict]:
        dataset = load_pubmedqa(split=split)
        output_path = self.config.output_dir / output_filename
        return self._write_scenarios(output_path, self._iter_pubmedqa(dataset, n_questions))

    def run_mmlu_med(
        self, n_questions: int, output_filename: str, split: str = "test"
    ) -> tuple[Path, dict]:
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
            print(f"\n{'=' * 60}")
            print(f"Q {idx + 1}/{limit} | query_id={query_id}")
            print(f"  {query[:100]}...")
            yield self._process_query(query_id, query, gold_answer)

    def _iter_mmlu_med(self, dataset, n_questions: int):
        limit = min(n_questions, len(dataset))
        for idx in range(limit):
            row = dataset[idx]
            query, options, gold_letter = format_mmlu_mcq(row)
            query_id = f"mmlu_med_{row['mmlu_subset']}_{idx}"
            gold_answer = options[gold_letter]
            print(f"\n{'=' * 60}")
            print(f"Q {idx + 1}/{limit} | query_id={query_id} | gold={gold_letter}")
            print(f"  {row['question'][:100]}...")
            yield self._process_query(query_id, query, gold_answer)

    def _write_scenarios(self, output_path: Path, scenarios) -> tuple[Path, dict]:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {"total": 0, "with_conflict": 0, "IC": 0, "CM": 0, "IM": 0}

        with output_path.open("w", encoding="utf-8") as handle:
            for scenario in scenarios:
                handle.write(json.dumps(scenario, ensure_ascii=False) + "\n")
                stats["total"] += 1
                if scenario["conflicts"]:
                    stats["with_conflict"] += 1
                for conflict in scenario["conflicts"]:
                    stats[conflict["type"]] += 1

        print(f"\n{'=' * 60}")
        print(f"Pipeline finished -> {output_path}")
        print(
            f"Scenarios: {stats['total']} total, "
            f"{stats['with_conflict']} with at least one conflict"
        )
        print(f"  IC={stats['IC']}  CM={stats['CM']}  IM={stats['IM']}")
        return output_path, stats


def pubmedqa_gold_answer(row) -> str:
    if "long_answer" in row and row["long_answer"]:
        return row["long_answer"]
    if "final_decision" in row and row["final_decision"]:
        return row["final_decision"]
    if "answer" in row and row["answer"]:
        return row["answer"]
    return ""
