"""Agnostic NLI conflict detection over normalized evidence elements."""

from __future__ import annotations

from itertools import combinations
import logging
from typing import Any

from .nli import NLIClassifier


Element = dict[str, Any]
LOG = logging.getLogger(__name__)


def normalize_retrieved_elements(retrieval_results: dict) -> list[Element]:
    # Stage 1 – collect all snippets, keyed by document_id for deduplication.
    # When the same document is returned by multiple retrievers we keep the
    # entry with the highest retrieval score and record every retriever that
    # found it so that provenance is preserved.
    best_by_doc: dict[str, Element] = {}   # document_id -> best element so far
    total_raw = 0

    for retriever, snippets in retrieval_results.items():
        for snippet in snippets:
            total_raw += 1
            rank = int(snippet.get("rank", 0))
            score = float(snippet.get("score", 0.0))
            document_id = snippet.get("document_id")

            if document_id in best_by_doc:
                # Duplicate – merge retriever info, keep highest-scored version
                existing = best_by_doc[document_id]
                if retriever not in existing["retrievers"]:
                    existing["retrievers"].append(retriever)
                if score > existing["score"]:
                    existing.update(
                        {
                            "_element_id": f"retrieved:{retriever}:{rank}:{document_id}",
                            "rank": rank,
                            "score": score,
                            "title": snippet.get("title", ""),
                            "text": snippet.get("text", ""),
                            "source_corpus": snippet.get("source_corpus", ""),
                            "retriever": retriever,
                            "content": snippet.get("content"),
                        }
                    )
            else:
                best_by_doc[document_id] = {
                    "_element_id": f"retrieved:{retriever}:{rank}:{document_id}",
                    "source_type": "retrieved",
                    "rank": rank,
                    "score": score,
                    "document_id": document_id,
                    "title": snippet.get("title", ""),
                    "text": snippet.get("text", ""),
                    "source_corpus": snippet.get("source_corpus", ""),
                    "retriever": retriever,
                    "retrievers": [retriever],
                    "content": snippet.get("content"),
                }

    elements = list(best_by_doc.values())
    duplicates_removed = total_raw - len(elements)
    LOG.info(
        "Normalized retrieved elements: raw=%s unique=%s duplicates_removed=%s",
        total_raw,
        len(elements),
        duplicates_removed,
    )
    return elements


def normalize_parametric_elements(probes: list[str], model_name: str) -> list[Element]:
    elements = []
    for index, text in enumerate(probes, start=1):
        elements.append(
            {
                "_element_id": f"parametric:{model_name}:{index}",
                "source_type": "parametric",
                "rank": index,
                "score": 0.0,
                "document_id": f"probe_{index}",
                "title": f"Parametric probe {index}",
                "text": text,
                "source_corpus": "parametric",
                "retriever": model_name,
                "content": None,
            }
        )
    LOG.info("Normalized parametric elements: count=%s", len(elements))
    return elements


def strip_internal_fields(element: Element) -> Element:
    return {key: value for key, value in element.items() if not key.startswith("_")}


def infer_conflict_type(left: Element, right: Element) -> str:
    source_types = {left["source_type"], right["source_type"]}
    if source_types == {"retrieved"}:
        return "IC"
    if source_types == {"parametric"}:
        return "IM"
    return "CM"


def detect_conflicts(query_id: str, elements: list[Element], nli: NLIClassifier) -> list[dict]:
    pairs = list(combinations(range(len(elements)), 2))
    LOG.info("Detecting conflicts: query_id=%s elements=%s pairs=%s", query_id, len(elements), len(pairs))
    nli_inputs = [(elements[i]["text"], elements[j]["text"]) for i, j in pairs]
    results = nli.classify_pairs(nli_inputs) if nli_inputs else []

    conflicts = []
    for pair_index, (left_index, right_index) in enumerate(pairs):
        result = results[pair_index]
        if not nli.is_contradiction(result):
            continue

        left = elements[left_index]
        right = elements[right_index]
        conflict_type = infer_conflict_type(left, right)
        conflicts.append(
            {
                "conflict_id": f"{query_id}_{conflict_type}_{len(conflicts) + 1:04d}",
                "type": conflict_type,
                "nli_label": result["label"].lower(),
                "nli_score": float(result["score"]),
                "conflictual_element": [
                    strip_internal_fields(left),
                    strip_internal_fields(right),
                ],
            }
        )

    counts = {"IC": 0, "CM": 0, "IM": 0}
    for conflict in conflicts:
        counts[conflict["type"]] += 1
    LOG.info(
        "Conflict detection finished: query_id=%s total=%s pairs=%s IC=%s CM=%s IM=%s",
        query_id,
        len(conflicts),
        len(pairs),
        counts["IC"],
        counts["CM"],
        counts["IM"],
    )
    return conflicts


def non_conflictual_elements(elements: list[Element], conflicts: list[dict]) -> list[Element]:
    conflictual_keys = {
        element_key(element)
        for conflict in conflicts
        for element in conflict["conflictual_element"]
    }
    if not conflictual_keys:
        return [strip_internal_fields(element) for element in elements]

    output = []
    for element in elements:
        if element_key(element) not in conflictual_keys:
            output.append(strip_internal_fields(element))
    return output


def element_key(element: Element) -> tuple:
    return (
        element.get("source_type"),
        element.get("rank"),
        element.get("document_id"),
        element.get("retriever"),
        element.get("text"),
    )
