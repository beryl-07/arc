"""Enrich SFT questions by executing the eight ICR strategies.

The input JSONL file is expected to contain one question per line. The script
builds one prompt per strategy, runs a causal language model in batches, checks
each response against the gold answer, and writes the original record enriched
with a ``valid_strategies`` field.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOG = logging.getLogger(__name__)
ABSTENTION_PHRASE = "The documents cannot answer this question."
STRATEGY_ORDER = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_INPUT_PATH = Path("data/sft_train.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/enriched_sft_train.jsonl")
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_MAX_INPUT_TOKENS = 4096
DEFAULT_MAX_DOCUMENT_CHARS = 1200
DEFAULT_MAX_TOTAL_DOCUMENT_CHARS = 6000
DEFAULT_QUESTION_BATCH_SIZE = 2


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    prompt_template: str
    expects_multiple_answers: bool = False
    can_abstain: bool = False


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        "S1",
        """You are resolving a knowledge conflict using Strategy S1: Identifying Correct Information.

Reference documents:
{documents}

Target question:
{question}

Apply Strategy S1 to resolve any conflict between the reference documents and the model's internal knowledge.
Determine which information is the most accurate and reliable, and use it to answer the question.
Output only a short answer, usually a phrase without explanation.

Answer:""",
    ),
    StrategySpec(
        "S2",
        """You are resolving a knowledge conflict using Strategy S2: Prioritizing External Knowledge.

Reference documents:
{documents}

Target question:
{question}

Apply Strategy S2.
When the reference documents and internal knowledge conflict, prioritize the information contained in the reference documents.
Output only a short answer, usually a phrase without explanation.

Answer:""",
    ),
    StrategySpec(
        "S3",
        """You are resolving a knowledge conflict using Strategy S3: Generating Corresponding Responses Respectively.

Reference documents:
{documents}

Target question:
{question}

Apply Strategy S3.
Provide two short answers:
1. one based on the reference documents;
2. one based on the model's internal knowledge.

Keep the two answers distinct.
Output only the two short answers, without explanation.

Answers:""",
        expects_multiple_answers=True,
    ),
    StrategySpec(
        "S4",
        """You are resolving a knowledge conflict using Strategy S4: Identifying Correct Information.

Reference documents:
{documents}

Target question:
{question}

Apply Strategy S4 to resolve conflicts among the reference documents.
Determine which information is the most accurate and reliable among the conflicting documents, and use it to answer the question.
Output only a short answer, usually a phrase without explanation.

Answer:""",
    ),
    StrategySpec(
        "S5",
        """You are resolving a knowledge conflict using Strategy S5: Prioritizing High-Frequency Information.

Reference documents:
{documents}

Target question:
{question}

Apply Strategy S5.
When the reference documents contain conflicting information, identify which information appears most frequently and prioritize it when answering.
Output only a short answer, usually a phrase without explanation.

Answer:""",
    ),
    StrategySpec(
        "S6",
        """You are resolving a knowledge conflict using Strategy S6: Generating Corresponding Responses Respectively.

Reference documents:
{documents}

Target question:
{question}

Apply Strategy S6.
If the reference documents contain conflicting information, provide a separate short answer corresponding to each conflicting position found in the documents.
Output only the short answers, without explanation.

Answers:""",
        expects_multiple_answers=True,
    ),
    StrategySpec(
        "S7",
        """You are resolving a knowledge conflict using Strategy S7: Prioritizing External Knowledge.

Reference documents:
{documents}

Target question:
{question}

Apply Strategy S7.
Answer the question using the reference documents.
If the reference documents do not contain sufficient information to answer the question, output exactly:
"The documents cannot answer this question."
Output only the answer.

Answer:""",
        can_abstain=True,
    ),
    StrategySpec(
        "S8",
        """You are resolving a knowledge conflict using Strategy S8: External Knowledge with Internal-Knowledge Fallback.

Reference documents:
{documents}

Target question:
{question}

Apply Strategy S8.
Use the reference documents to answer the question when they contain relevant information.
If the reference documents do not contain the relevant information, use the model's internal knowledge instead.
Output only a short answer, usually a phrase without explanation.

Answer:""",
    ),
)


class StrategyEnricher:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int,
        max_input_tokens: int,
        max_document_chars: int,
        max_total_document_chars: int,
        trust_remote_code: bool,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.max_document_chars = max_document_chars
        self.max_total_document_chars = max_total_document_chars
        self.trust_remote_code = trust_remote_code
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            LOG.info("Loading tokenizer: model=%s", self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            tokenizer.padding_side = "left"
            tokenizer.truncation_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            LOG.info("Loading model: model=%s dtype=%s", self.model_name, dtype)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=dtype,
                device_map="auto",
            )
            self._model.eval()
        return self._model

    def build_documents_text(self, record: dict[str, Any]) -> str:
        documents = record.get("documents")
        if isinstance(documents, list) and documents:
            formatted = self._format_documents_list(documents)
            return formatted or "No documents provided."

        reconstructed = self._reconstruct_documents(record)
        if reconstructed:
            return self._format_documents_list(reconstructed)
        return "No documents provided."

    def _reconstruct_documents(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        def add_element(element: Any) -> None:
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
            documents.append(
                {
                    "id": doc_id,
                    "text": text,
                    "title": title,
                    "retriever": element.get("retriever"),
                    "source_type": element.get("source_type"),
                }
            )

        for element in record.get("non_conflictual_elements", []) or []:
            add_element(element)

        for conflict in record.get("conflicts", []) or []:
            for element in conflict.get("conflictual_element", []) or []:
                add_element(element)

        return documents

    def _format_documents_list(self, documents: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        total_chars = 0
        for idx, document in enumerate(documents, start=1):
            if total_chars >= self.max_total_document_chars:
                break

            doc_id = str(document.get("id") or document.get("document_id") or idx)
            title = str(document.get("title") or "").strip()
            text = str(document.get("text") or document.get("content") or "").strip()
            if self.max_document_chars > 0 and len(text) > self.max_document_chars:
                text = text[: self.max_document_chars].rstrip() + "..."

            header = f"[{idx}] {doc_id}"
            if title:
                header += f" | {title}"
            chunk = f"{header}\n{text}".strip()
            chunk_len = len(chunk)
            if total_chars + chunk_len > self.max_total_document_chars:
                remaining = self.max_total_document_chars - total_chars
                if remaining <= 0:
                    break
                chunk = chunk[:remaining].rstrip()
                lines.append(chunk)
                break

            lines.append(chunk)
            total_chars += chunk_len + 1

        return "\n\n".join(lines).strip()

    def build_prompts(self, record: dict[str, Any]) -> list[tuple[str, str]]:
        documents_text = self.build_documents_text(record)
        question = str(record.get("query", "")).strip()
        prompts: list[tuple[str, str]] = []
        for strategy in STRATEGIES:
            prompt = strategy.prompt_template.format(documents=documents_text, question=question)
            prompts.append((strategy.strategy_id, prompt))
        return prompts

    def generate_batch(self, prompts: list[str]) -> list[str]:
        tokenizer = self.tokenizer
        model = self.model
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **encoded,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = encoded["input_ids"].shape[1]
        responses: list[str] = []
        for output in outputs:
            generated = output[prompt_len:]
            responses.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
        return responses

    def generate_with_fallback(self, prompts: list[str]) -> list[str]:
        try:
            return self.generate_batch(prompts)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("Batch generation failed for %s prompts; falling back to individual calls: %s", len(prompts), exc)
            responses: list[str] = []
            for index, prompt in enumerate(prompts):
                try:
                    responses.extend(self.generate_batch([prompt]))
                except Exception as prompt_exc:  # noqa: BLE001
                    LOG.exception("Generation failed for prompt %s: %s", index, prompt_exc)
                    responses.append("")
            return responses


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def parse_options(query: str) -> dict[str, str]:
    pattern = re.compile(r"(?ms)^\s*([A-D])\.\s*(.*?)(?=^\s*[A-D]\.\s*|\Z)")
    options: dict[str, str] = {}
    for match in pattern.finditer(query):
        letter = match.group(1)
        text = re.sub(r"\s+", " ", match.group(2)).strip()
        if text:
            options[letter] = text
    return options


def extract_answer_letter_from_query(query: str, gold_answer: Any) -> str | None:
    """Infer a choice letter by matching the gold answer text against options."""

    if not isinstance(gold_answer, str):
        return None

    normalized_gold = normalize_text(gold_answer)
    if not normalized_gold:
        return None

    options = parse_options(query)
    for letter, option_text in options.items():
        if normalize_text(option_text) == normalized_gold:
            return letter
    return None


def get_gold_letter(question_data: dict[str, Any]) -> str | None:
    """Extract the gold letter from either string or list gold-answer formats."""

    gold_answer = question_data.get("gold_answer", "")

    if isinstance(gold_answer, list):
        if not gold_answer:
            return None
        first = str(gold_answer[0]).strip().upper()
        return first or None

    if isinstance(gold_answer, str):
        letters = extract_choice_letters(gold_answer)
        if letters:
            return letters[0]

    return extract_answer_letter_from_query(
        str(question_data.get("query", "")),
        gold_answer,
    )


def resolve_gold_reference(record: dict[str, Any]) -> tuple[str | None, str, bool]:
    gold_answer = record.get("gold_answer", "")
    gold_letter = get_gold_letter(record)

    if isinstance(gold_answer, list):
        raw_gold = " ".join(str(item).strip() for item in gold_answer if str(item).strip())
    else:
        raw_gold = str(gold_answer).strip()

    if normalize_text(raw_gold) == normalize_text("ABSTENTION"):
        return None, raw_gold, True

    if gold_letter and re.fullmatch(r"[ABCD]", gold_letter.upper()):
        return gold_letter.upper(), raw_gold or gold_letter.upper(), False

    if isinstance(gold_answer, str):
        query = str(record.get("query", ""))
        options = parse_options(query)
        normalized_gold = normalize_text(raw_gold)
        for letter, option_text in options.items():
            normalized_option = normalize_text(option_text)
            if normalized_gold == normalized_option:
                return letter, option_text, False

    return None, raw_gold, False


def extract_choice_letters(text: str) -> list[str]:
    """Extract explicit multiple-choice letters from a model response.

    The extractor is intentionally strict: it only accepts answers that are
    clearly formatted as a choice label or a short labeled answer. This avoids
    treating ordinary prose like "A patient..." as a valid answer.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        lines = [text.strip()]

    patterns = (
        re.compile(r"^(?:answer(?: is)?|final answer|option|choice)\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\].:!\-]?\s*$", re.IGNORECASE),
        re.compile(r"^(?:\d+\s*[\).:-]\s*)?[\(\[]?\s*([ABCD])\s*[\)\].:!\-]?\s*$", re.IGNORECASE),
    )

    letters: list[str] = []
    for line in lines:
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                letters.append(match.group(1).upper())
                break

    if letters:
        return letters

    inline_pattern = re.compile(
        r"(?:^|\n)\s*(?:answer(?: is)?|final answer|option|choice)\s*[:\-]?\s*[\(\[]?\s*([ABCD])\s*[\)\].:!\-]?(?:\s|$)",
        re.IGNORECASE,
    )
    return [match.group(1).upper() for match in inline_pattern.finditer(text)]


def matches_gold(text: str, gold_letter: str | None, gold_text: str) -> bool:
    letters = extract_choice_letters(text)
    if gold_letter:
        return bool(letters) and letters[0] == gold_letter

    normalized_text = normalize_text(text)
    normalized_gold = normalize_text(gold_text)
    if not normalized_gold:
        return False
    return normalized_gold == normalized_text


def evaluate_response(
    strategy_id: str,
    response: str,
    gold_letter: str | None,
    gold_text: str,
    gold_is_abstention: bool,
) -> bool:
    if strategy_id == "S7" and ABSTENTION_PHRASE in response:
        return gold_is_abstention

    if strategy_id in {"S3", "S6"}:
        if gold_letter and gold_letter in extract_choice_letters(response):
            return True
        if gold_letter:
            return False
        return matches_gold(response, gold_letter, gold_text)

    return matches_gold(response, gold_letter, gold_text)


def chunked(iterable: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [iterable[index : index + size] for index in range(0, len(iterable), size)]


def enrich_records(
    records: list[dict[str, Any]],
    enricher: StrategyEnricher,
    output_path: Path,
    start_index: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    total = len(records)
    remaining = records[start_index:]
    batches = chunked(remaining, batch_size)

    mode = "a" if start_index > 0 else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode, encoding="utf-8") as handle:
        with tqdm(total=len(remaining), desc="Enriching questions", unit="question") as progress:
            for batch in batches:
                batch_prompts: list[str] = []
                batch_meta: list[tuple[int, str]] = []
                batch_payloads: list[dict[str, Any]] = []

                for record in batch:
                    gold_letter, gold_text, gold_is_abstention = resolve_gold_reference(record)
                    prompts = enricher.build_prompts(record)
                    batch_payloads.append(
                        {
                            "record": record,
                            "gold_letter": gold_letter,
                            "gold_text": gold_text,
                            "gold_is_abstention": gold_is_abstention,
                        }
                    )
                    for strategy_id, prompt in prompts:
                        batch_prompts.append(prompt)
                        batch_meta.append((len(batch_payloads) - 1, strategy_id))

                responses = enricher.generate_with_fallback(batch_prompts)

                for payload in batch_payloads:
                    payload["valid_strategies"] = []

                for response, (payload_index, strategy_id) in zip(responses, batch_meta, strict=False):
                    payload = batch_payloads[payload_index]
                    record = payload["record"]
                    try:
                        is_correct = evaluate_response(
                            strategy_id,
                            response,
                            payload["gold_letter"],
                            payload["gold_text"],
                            payload["gold_is_abstention"],
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.exception(
                            "Evaluation failed: query_id=%s strategy=%s error=%s",
                            record.get("query_id", ""),
                            strategy_id,
                            exc,
                        )
                        is_correct = False

                    if is_correct:
                        payload["valid_strategies"].append(strategy_id)

                for payload in batch_payloads:
                    enriched = dict(payload["record"])
                    enriched["valid_strategies"] = payload["valid_strategies"]
                    processed.append(enriched)
                    handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    handle.flush()

                progress.update(len(batch))

    if len(processed) != total - start_index:
        raise RuntimeError("Processed record count mismatch.")
    return processed


def summarize(records: list[dict[str, Any]]) -> None:
    total = len(records)
    distribution = Counter(len(record.get("valid_strategies", [])) for record in records)

    print(f"\nTotal questions processed: {total}")
    print("Distribution of valid strategies per question:")
    for count in range(0, 9):
        suffix = "s" if count != 1 else ""
        print(f"  {count} strategy{suffix}: {distribution.get(count, 0)}")
    print(f"  2+ strategies: {sum(value for key, value in distribution.items() if key >= 2)}")

    print("\nSample questions:")
    for record in records[:5]:
        print(f"  {record.get('query_id', '')}: {record.get('valid_strategies', [])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the eight ICR strategies over SFT questions and keep the winning strategies."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input JSONL file. Defaults to data/sft_train.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL file. Defaults to data/enriched_sft_train.jsonl.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name or local path.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--max-document-chars", type=int, default=DEFAULT_MAX_DOCUMENT_CHARS)
    parser.add_argument("--max-total-document-chars", type=int, default=DEFAULT_MAX_TOTAL_DOCUMENT_CHARS)
    parser.add_argument("--question-batch-size", type=int, default=DEFAULT_QUESTION_BATCH_SIZE)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing output file and rebuild from scratch.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        help="Disable loading custom model code from the Hugging Face repository.",
    )
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    records = read_jsonl(args.input_path)
    output_path = args.output_path

    if args.overwrite:
        start_index = 0
        if output_path.exists():
            LOG.info("Overwriting existing output file: %s", output_path)
    else:
        start_index = count_jsonl_lines(output_path)
        if start_index > len(records):
            raise ValueError(
                f"Existing output file has {start_index} lines, which is more than the {len(records)} input records."
            )
        if start_index > 0:
            LOG.info("Resuming from existing output file: %s completed records", start_index)

    enricher = StrategyEnricher(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        max_document_chars=args.max_document_chars,
        max_total_document_chars=args.max_total_document_chars,
        trust_remote_code=args.trust_remote_code,
    )

    enriched_records = enrich_records(
        records=records,
        enricher=enricher,
        output_path=output_path,
        start_index=start_index,
        batch_size=max(1, args.question_batch_size),
    )

    if start_index > 0:
        existing = read_jsonl(output_path)
        final_records = existing
    else:
        final_records = enriched_records

    summarize(final_records)
    LOG.info("Wrote enriched dataset to %s", output_path)


if __name__ == "__main__":
    main()
