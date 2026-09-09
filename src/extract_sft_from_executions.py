#!/usr/bin/env python3
"""
Extrait les exemples SFT à partir des valid_strategies et des executions structurées.
Chaque assistant contient le JSON complet (strategy, response, justification).
"""

import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import read_jsonl, write_jsonl

INPUT_PATH = PROJECT_ROOT / "data" / "enriched_sft_train.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "sft_execution_data.jsonl"

# Prompts d'exécution (courts, pour le user prompt)
STRATEGY_PROMPTS = {
    "S1": "Apply Strategy S1: Identifying Correct Information. Resolve conflicts between documents and internal knowledge.",
    "S2": "Apply Strategy S2: Prioritizing External Knowledge. Prioritize document information over internal knowledge.",
    "S3": "Apply Strategy S3: Generating Corresponding Responses Respectively. Provide two answers: one from documents, one from internal knowledge.",
    "S4": "Apply Strategy S4: Identifying Correct Information Among Documents. Resolve conflicts between documents.",
    "S5": "Apply Strategy S5: Prioritizing High-Frequency Information. Identify the most frequent information in documents.",
    "S6": "Apply Strategy S6: Generating Corresponding Responses from Conflicting Documents. Provide a separate answer for each conflicting document.",
    "S7": "Apply Strategy S7: Prioritizing External Knowledge with Abstention. Answer using documents. If insufficient, output 'ABSTENTION'.",
    "S8": "Apply Strategy S8: External Knowledge with Internal-Knowledge Fallback. Use documents if possible, otherwise use internal knowledge.",
}

MAX_DOCUMENT_CHARS = 500
MAX_DOCUMENTS = 3


def reconstruct_documents(record: dict) -> list[dict]:
    """Reconstruit la liste de documents à partir de conflicts et non_conflictual_elements."""
    documents: list[dict] = []
    seen_ids: set[str] = set()

    def add_element(element):
        if not isinstance(element, dict):
            return
        doc_id = str(
            element.get("id")
            or element.get("document_id")
            or element.get("_element_id")
            or element.get("title")
            or len(documents)
        )
        if doc_id in seen_ids:
            return
        seen_ids.add(doc_id)
        text = element.get("text") or element.get("content") or ""
        title = element.get("title") or element.get("source_corpus") or ""
        documents.append({"id": doc_id, "text": text, "title": title})

    for element in record.get("non_conflictual_elements", []) or []:
        add_element(element)

    for conflict in record.get("conflicts", []) or []:
        for element in conflict.get("conflictual_element", []) or []:
            add_element(element)

    return documents


def build_user_prompt(query: str, documents: list, strategy_id: str) -> str:
    """Construit le prompt utilisateur avec la question, les documents et la stratégie."""
    if documents:
        doc_lines = []
        for d in documents[:MAX_DOCUMENTS]:
            text = d.get("text", "")[:MAX_DOCUMENT_CHARS]
            title = d.get("title", "")
            header = title if title else d.get("id", "")
            doc_lines.append(f"- [{header}] {text}")
        docs_text = "\n".join(doc_lines)
    else:
        docs_text = "No documents provided."
    return (
        f"Question: {query}\n\n"
        f"Documents:\n{docs_text}\n\n"
        f"Strategy: {strategy_id} - {STRATEGY_PROMPTS.get(strategy_id, '')}\n\n"
        "Apply the strategy. Output a strict JSON with strategy, response, and justification."
    )


def main():
    print(f"📂 Chargement de {INPUT_PATH}...")
    records = read_jsonl(INPUT_PATH)
    print(f"   Total enregistrements: {len(records)}")

    examples = []
    strategy_counter: Counter = Counter()
    question_count = 0

    for rec in records:
        valid = rec.get("valid_strategies", [])
        if not valid:
            continue
        question_count += 1
        executions = rec.get("executions", {})
        query = rec.get("query", "")

        # Reconstruit les documents à partir de conflicts / non_conflictual_elements
        documents = rec.get("documents") or reconstruct_documents(rec)

        for sid in valid:
            exec_data = executions.get(sid, {})
            # On garde la réponse brute (JSON complet)
            structured_response = exec_data.get("response", "")

            if not structured_response:
                continue

            # Construction de l'exemple SFT
            example = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a medical assistant. Apply the specified resolution "
                            "strategy strictly. Output your answer as a valid JSON with "
                            "fields: strategy, response, and justification."
                        ),
                    },
                    {
                        "role": "user",
                        "content": build_user_prompt(query, documents, sid),
                    },
                    {
                        "role": "assistant",
                        "content": structured_response.strip(),
                    },
                ]
            }
            examples.append(example)
            strategy_counter[sid] += 1

    print(f"\n✅ Questions avec au moins une stratégie valide: {question_count}")
    print(f"✅ Total d'exemples SFT générés: {len(examples)}")
    print("\n📈 Répartition par stratégie:")
    for s, c in sorted(strategy_counter.items()):
        print(f"  {s}: {c}")

    write_jsonl(OUTPUT_PATH, examples)
    print(f"\n📁 Dataset SFT sauvegardé: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
