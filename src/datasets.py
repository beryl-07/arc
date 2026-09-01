"""Dataset loading and question formatting helpers."""

from __future__ import annotations

import logging


LOG = logging.getLogger(__name__)

MMIU_MED_SUBSETS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]

# Alias kept for backwards-compat
MMIU_MED_SUBSETS = MMLU_MED_SUBSETS = MMIU_MED_SUBSETS


def load_pubmedqa(split: str = "train"):
    from datasets import load_dataset

    LOG.info("Loading PubMedQA dataset: split=%s", split)
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
    LOG.info("PubMedQA loaded: split=%s count=%s", split, len(dataset))
    return dataset


def load_mmlu_med(split: str = "test"):
    from datasets import concatenate_datasets, load_dataset

    LOG.info("Loading MMLU-Med dataset: split=%s", split)
    subsets = []
    for name in MMLU_MED_SUBSETS:
        LOG.info("Loading MMLU subset: %s", name)
        dataset = load_dataset("cais/mmlu", name, split=split)
        dataset = dataset.add_column("mmlu_subset", [name] * len(dataset))
        subsets.append(dataset)
    mmlu_med = concatenate_datasets(subsets)
    LOG.info("MMLU-Med loaded: split=%s count=%s", split, len(mmlu_med))
    return mmlu_med


def load_medqa_us(split: str = "test"):
    """Load MedQA-USMLE 4-option dataset from Hugging Face.

    HF repo: GBaker/MedQA-USMLE-4-options
    Columns: question, options (dict A-D), answer_idx (letter), answer (text)
    """
    from datasets import load_dataset

    LOG.info("Loading MedQA-US dataset: split=%s", split)
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    LOG.info("MedQA-US loaded: split=%s count=%s", split, len(dataset))
    return dataset


def load_medmcqa(split: str = "validation"):
    """Load MedMCQA dataset from Hugging Face.

    HF repo: openlifescienceai/medmcqa
    Columns: id, question, opa, opb, opc, opd, cop (0-3), subject_name, topic_name,
             explanation, choice_type
    """
    from datasets import load_dataset

    LOG.info("Loading MedMCQA dataset: split=%s", split)
    # Use the default configuration (plain 'medmcqa')
    dataset = load_dataset("openlifescienceai/medmcqa", split=split)
    LOG.info("MedMCQA loaded: split=%s count=%s", split, len(dataset))
    return dataset


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_mmlu_mcq(question_row) -> tuple[str, dict[str, str], str]:
    """Format an MMLU-Med row into (query_text, options_dict, gold_letter)."""
    letters = ["A", "B", "C", "D"]
    options = {letters[i]: choice for i, choice in enumerate(question_row["choices"])}
    options_text = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    query_text = f"{question_row['question']}\n{options_text}"
    gold_letter = letters[int(question_row["answer"])]
    return query_text, options, gold_letter


def format_medqa_us_mcq(question_row) -> tuple[str, dict[str, str], str]:
    """Format a MedQA-US row into (query_text, options_dict, gold_letter).

    The ``options`` column is a dict like {"A": "...", "B": "...", ...}.
    ``answer_idx`` holds the correct letter (e.g. "B").
    """
    options: dict[str, str] = question_row["options"]
    # Normalise to sorted letter order A-D
    ordered = dict(sorted(options.items()))
    options_text = "\n".join(f"{letter}. {text}" for letter, text in ordered.items())
    query_text = f"{question_row['question']}\n{options_text}"
    gold_letter = question_row["answer_idx"].strip().upper()
    return query_text, ordered, gold_letter


def format_medmcqa_mcq(question_row) -> tuple[str, dict[str, str], str]:
    """Format a MedMCQA row into (query_text, options_dict, gold_letter).

    Options are stored as opa, opb, opc, opd; ``cop`` is 0-indexed.
    """
    letters = ["A", "B", "C", "D"]
    options = {
        "A": str(question_row["opa"]),
        "B": str(question_row["opb"]),
        "C": str(question_row["opc"]),
        "D": str(question_row["opd"]),
    }
    options_text = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    query_text = f"{question_row['question']}\n{options_text}"
    gold_letter = letters[int(question_row["cop"])]
    return query_text, options, gold_letter
