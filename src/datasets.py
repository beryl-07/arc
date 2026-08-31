"""Dataset loading and question formatting helpers."""

from __future__ import annotations


MMLU_MED_SUBSETS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]


def load_pubmedqa(split: str = "train"):
    from datasets import load_dataset

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split=split)
    print(f"PubMedQA ({split}): {len(dataset)} questions loaded.")
    return dataset


def load_mmlu_med(split: str = "test"):
    from datasets import concatenate_datasets, load_dataset

    subsets = []
    for name in MMLU_MED_SUBSETS:
        dataset = load_dataset("cais/mmlu", name, split=split)
        dataset = dataset.add_column("mmlu_subset", [name] * len(dataset))
        subsets.append(dataset)
    mmlu_med = concatenate_datasets(subsets)
    print(
        f"MMLU-Med ({split}): {len(mmlu_med)} questions loaded "
        f"({', '.join(MMLU_MED_SUBSETS)})."
    )
    return mmlu_med


def format_mmlu_mcq(question_row) -> tuple[str, dict[str, str], str]:
    letters = ["A", "B", "C", "D"]
    options = {letters[i]: choice for i, choice in enumerate(question_row["choices"])}
    options_text = "\n".join(f"{letter}. {text}" for letter, text in options.items())
    query_text = f"{question_row['question']}\n{options_text}"
    gold_letter = letters[int(question_row["answer"])]
    return query_text, options, gold_letter
