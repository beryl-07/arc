"""SFT dataset construction helpers."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator


LOG = logging.getLogger(__name__)

STRATEGY_ORDER = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
ABSTENTION_TEXT = "The documents cannot answer this question."


# ---------------------------------------------------------------------------
# Strategy prompts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    name: str
    prompt: str


STRATEGY_SPECS: dict[str, StrategySpec] = {
    "S1": StrategySpec(
        strategy_id="S1",
        name="Identifying Correct Information",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S1: Identifying Correct Information.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S1 to resolve any conflict between the reference documents and the model's internal knowledge.\n"
            "Determine which information is the most accurate and reliable, and use it to answer the question.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
    "S2": StrategySpec(
        strategy_id="S2",
        name="Prioritizing External Knowledge",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S2: Prioritizing External Knowledge.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S2.\n"
            "When the reference documents and internal knowledge conflict, prioritize the information contained in the reference documents.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
    "S3": StrategySpec(
        strategy_id="S3",
        name="Generating Corresponding Responses Respectively",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S3: Generating Corresponding Responses Respectively.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S3.\n"
            "Provide two short answers:\n"
            "1. one based on the reference documents;\n"
            "2. one based on the model's internal knowledge.\n\n"
            "Keep the two answers distinct.\n"
            "Output only the two short answers, without explanation.\n\n"
            "Answers:"
        ),
    ),
    "S4": StrategySpec(
        strategy_id="S4",
        name="Identifying Correct Information Among Documents",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S4: Identifying Correct Information.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S4 to resolve conflicts among the reference documents.\n"
            "Determine which information is the most accurate and reliable among the conflicting documents, and use it to answer the question.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
    "S5": StrategySpec(
        strategy_id="S5",
        name="Prioritizing High-Frequency Information",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S5: Prioritizing High-Frequency Information.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S5.\n"
            "When the reference documents contain conflicting information, identify which information appears most frequently and prioritize it when answering.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
    "S6": StrategySpec(
        strategy_id="S6",
        name="Generating Corresponding Responses from Conflicting Documents",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S6: Generating Corresponding Responses Respectively.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S6.\n"
            "If the reference documents contain conflicting information, provide a separate short answer corresponding to each conflicting position found in the documents.\n"
            "Output only the short answers, without explanation.\n\n"
            "Answers:"
        ),
    ),
    "S7": StrategySpec(
        strategy_id="S7",
        name="Prioritizing External Knowledge with Abstention",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S7: Prioritizing External Knowledge.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S7.\n"
            "Answer the question using the reference documents.\n"
            "If the reference documents do not contain sufficient information to answer the question, output exactly:\n"
            "\"The documents cannot answer this question.\"\n\n"
            "Output only the answer.\n\n"
            "Answer:"
        ),
    ),
    "S8": StrategySpec(
        strategy_id="S8",
        name="External Knowledge with Internal-Knowledge Fallback",
        prompt=(
            "You are resolving a knowledge conflict using Strategy S8: External Knowledge with Internal-Knowledge Fallback.\n\n"
            "Reference documents:\n{documents}\n\n"
            "Target question:\n{question}\n\n"
            "Apply Strategy S8.\n"
            "Use the reference documents to answer the question when they contain relevant information.\n"
            "If the reference documents do not contain the relevant information, use the model's internal knowledge instead.\n"
            "Output only a short answer, usually a phrase without explanation.\n\n"
            "Answer:"
        ),
    ),
}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Invalid JSON object on line {line_number} of {path}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------


class StrategyRunner:
    """Load the local LLM once and reuse it for all eight strategy calls."""

    def __init__(self, model_name: str, temperature: float, max_new_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            LOG.info("Loading SFT model: model=%s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            LOG.info("SFT model loaded: model=%s", self.model_name)
        return self._model, self._tokenizer

    def generate(self, prompt: str) -> str:
        import torch

        model, tokenizer = self.load_model()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------


def flatten_scenario_documents(scenario: dict[str, Any]) -> list[dict[str, Any]]:
    raw_documents: list[dict[str, Any]] = []
    for conflict in scenario.get("conflicts", []):
        if not isinstance(conflict, dict):
            continue
        for element in conflict.get("conflictual_element", []):
            if isinstance(element, dict):
                raw_documents.append(element)
    for element in scenario.get("non_conflictual_elements", []):
        if isinstance(element, dict):
            raw_documents.append(element)

    seen: set[tuple[Any, ...]] = set()
    documents: list[dict[str, Any]] = []
    for raw in raw_documents:
        key = (
            raw.get("document_id"),
            raw.get("source_type"),
            raw.get("rank"),
            raw.get("retriever"),
            raw.get("text"),
        )
        if key in seen:
            continue
        seen.add(key)
        documents.append(
            {
                "document_id": raw.get("document_id", ""),
                "title": raw.get("title", ""),
                "text": raw.get("text", ""),
                "source_type": raw.get("source_type", ""),
                "rank": raw.get("rank"),
                "score": raw.get("score"),
                "source_corpus": raw.get("source_corpus", ""),
                "retriever": raw.get("retriever", ""),
                "content": raw.get("content"),
            }
        )
    return documents


def assign_document_ids(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, document in enumerate(documents, start=1):
        output.append({"id": f"D{index}", **document})
    return output


def render_documents(documents: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for document in documents:
        lines = [f"[Document {document['id']}]"]
        title = str(document.get("title", "")).strip()
        if title:
            lines.append(f"Title: {title}")
        text = str(document.get("text", "")).strip()
        lines.append(f"Text: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_strategy_prompt(strategy_id: str, question: str, documents: list[dict[str, Any]]) -> str:
    if strategy_id not in STRATEGY_SPECS:
        raise KeyError(f"Unknown strategy_id: {strategy_id}")
    spec = STRATEGY_SPECS[strategy_id]
    return spec.prompt.format(question=question, documents=render_documents(documents))


def build_strategy_messages(strategy_id: str, question: str, documents: list[dict[str, Any]]) -> list[dict[str, str]]:
    prompt = build_strategy_prompt(strategy_id, question, documents)
    return [
        {
            "role": "system",
            "content": "You are a dataset construction assistant. Follow the assigned strategy exactly.",
        },
        {"role": "user", "content": prompt},
    ]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


_LEADING_LABEL_RE = re.compile(r"^\s*(answer|answers)\s*:\s*", re.IGNORECASE)
_LEADING_BULLET_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_LEADING_OPTION_RE = re.compile(r"^\s*[A-H]\s*[\.\):\-]\s*")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def extract_answer_candidates(response: str) -> list[str]:
    response = response.strip()
    if not response:
        return []

    candidates: list[str] = []
    for chunk in re.split(r"[\n;]+", response):
        item = chunk.strip()
        if not item:
            continue
        item = _LEADING_LABEL_RE.sub("", item)
        item = _LEADING_BULLET_RE.sub("", item)
        item = _LEADING_OPTION_RE.sub("", item)
        if item:
            candidates.append(item)

    if not candidates:
        candidates.append(response)
    return candidates


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("\n", " ")
    text = _LEADING_LABEL_RE.sub("", text)
    text = _LEADING_BULLET_RE.sub("", text)
    text = _LEADING_OPTION_RE.sub("", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = _ARTICLES_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct_response(response: str, gold_answer: str, strategy_id: str) -> bool:
    gold_norm = normalize_answer(gold_answer)
    if not gold_norm:
        return False

    for candidate in extract_answer_candidates(response):
        candidate_norm = normalize_answer(candidate)
        if not candidate_norm:
            continue

        if candidate_norm == gold_norm:
            return True
        if gold_norm in candidate_norm or candidate_norm in gold_norm:
            return True

        if strategy_id in {"S3", "S6"}:
            if any(
                gold_norm == normalize_answer(piece)
                for piece in re.split(r"[\n;]+", candidate)
                if piece.strip()
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def build_question_record(
    scenario: dict[str, Any],
    strategy_outputs: dict[str, dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    question_id = str(scenario.get("query_id", ""))
    question = str(scenario.get("query", ""))
    gold_answer = str(scenario.get("gold_answer", ""))
    documents = assign_document_ids(flatten_scenario_documents(scenario))

    valid_strategies = [strategy_id for strategy_id in STRATEGY_ORDER if strategy_outputs[strategy_id]["correct"]]

    return {
        "question_id": question_id,
        "query": question,
        "gold_answer": gold_answer,
        "input": {
            "question": question,
            "documents": documents,
        },
        "scenario": scenario,
        "executions": strategy_outputs,
        "valid_strategies": valid_strategies,
        "model_name": model_name,
    }


def build_sft_examples(question_record: dict[str, Any]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    question_id = str(question_record["question_id"])
    question = str(question_record["query"])
    documents = list(question_record["input"]["documents"])
    scenario = question_record["scenario"]
    valid_strategies = set(question_record["valid_strategies"])
    executions = question_record["executions"]

    for strategy_id in STRATEGY_ORDER:
        if strategy_id not in valid_strategies:
            continue
        execution = executions[strategy_id]
        prompt = build_strategy_prompt(strategy_id, question, documents)
        examples.append(
            {
                "question_id": question_id,
                "strategy_id": strategy_id,
                "valid_strategies": question_record["valid_strategies"],
                "input": {
                    "question": question,
                    "documents": documents,
                    "strategy": strategy_id,
                },
                "scenario": scenario,
                "prompt": prompt,
                "output": {
                    "answer": execution["answer"],
                },
                "execution": execution,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a dataset construction assistant. Follow the assigned strategy exactly.",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": execution["answer"]},
                ],
            }
        )
    return examples


def _stable_split_bucket(question_id: str, seed: int) -> float:
    digest = sha256(f"{seed}:{question_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def split_question_ids(
    question_ids: list[str],
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[str]]:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")
    if min(train_ratio, validation_ratio, test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")

    split_map = {"train": [], "validation": [], "test": []}
    for question_id in sorted(set(question_ids)):
        bucket = _stable_split_bucket(question_id, seed)
        if bucket < train_ratio:
            split_map["train"].append(question_id)
        elif bucket < train_ratio + validation_ratio:
            split_map["validation"].append(question_id)
        else:
            split_map["test"].append(question_id)
    return split_map


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def load_scenarios(path: Path, n_questions: int | None = None) -> list[dict[str, Any]]:
    scenarios = load_jsonl(path)
    if n_questions is not None:
        scenarios = scenarios[:n_questions]
    return scenarios


def run_sft_pipeline(
    scenarios_path: Path,
    output_dir: Path,
    model_name: str,
    n_questions: int | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    scenarios = load_scenarios(scenarios_path, n_questions=n_questions)
    runner = StrategyRunner(model_name=model_name, temperature=temperature, max_new_tokens=max_new_tokens)

    question_records: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios, start=1):
        question_id = str(scenario.get("query_id", f"question_{index}"))
        question = str(scenario.get("query", ""))
        gold_answer = str(scenario.get("gold_answer", ""))
        documents = assign_document_ids(flatten_scenario_documents(scenario))
        LOG.info("Processing question: question_id=%s strategy_count=%s", question_id, len(STRATEGY_ORDER))

        strategy_outputs: dict[str, dict[str, Any]] = {}
        for strategy_id in STRATEGY_ORDER:
            prompt = build_strategy_prompt(strategy_id, question, documents)
            answer = runner.generate(prompt)
            correct = is_correct_response(answer, gold_answer, strategy_id)
            strategy_outputs[strategy_id] = {
                "strategy_id": strategy_id,
                "prompt": prompt,
                "answer": answer,
                "correct": correct,
                "score": 1.0 if correct else 0.0,
                "gold_answer": gold_answer,
            }
            LOG.info(
                "Strategy evaluated: question_id=%s strategy_id=%s correct=%s preview=%r",
                question_id,
                strategy_id,
                correct,
                answer[:80],
            )

        question_records.append(build_question_record(scenario, strategy_outputs, model_name=model_name))

    question_ids = [record["question_id"] for record in question_records]
    split_map = split_question_ids(
        question_ids,
        seed=seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )
    question_to_split = {
        question_id: split_name
        for split_name, ids in split_map.items()
        for question_id in ids
    }

    split_examples: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for question_record in question_records:
        split_name = question_to_split[question_record["question_id"]]
        split_examples[split_name].extend(build_sft_examples(question_record))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "questions.jsonl", question_records)
    write_jsonl(output_dir / "valid_strategies.jsonl", question_records)
    for split_name, examples in split_examples.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", examples)

    summary = {
        "scenarios": len(question_records),
        "train_examples": len(split_examples["train"]),
        "validation_examples": len(split_examples["validation"]),
        "test_examples": len(split_examples["test"]),
        "questions_with_at_least_one_valid_strategy": sum(1 for record in question_records if record["valid_strategies"]),
    }
    LOG.info(
        "SFT pipeline finished: scenarios=%s train=%s validation=%s test=%s",
        summary["scenarios"],
        summary["train_examples"],
        summary["validation_examples"],
        summary["test_examples"],
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the SFT dataset by executing 8 strategies per scenario.")
    parser.add_argument("scenarios", type=Path, help="Input scenario JSONL file.")
    parser.add_argument("output_dir", type=Path, help="Output directory for SFT artifacts.")
    parser.add_argument("--n-questions", type=int, default=None, help="Optional limit on the number of scenarios.")
    parser.add_argument("--model", default=None, help="Model name to use for execution. Defaults to PKE_MODEL.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from .config import load_config

    config = load_config()
    model_name = args.model or config.pke_model
    temperature = args.temperature if args.temperature is not None else config.pke_temperature
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config.pke_max_new_tokens

    LOG.info(
        "Starting SFT construction: scenarios=%s output_dir=%s model=%s",
        args.scenarios,
        args.output_dir,
        model_name,
    )
    run_sft_pipeline(
        scenarios_path=args.scenarios,
        output_dir=args.output_dir,
        model_name=model_name,
        n_questions=args.n_questions,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


if __name__ == "__main__":
    main()
