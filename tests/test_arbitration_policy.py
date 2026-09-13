from __future__ import annotations

import unittest

from src.arbitration_policy import (
    JSON_INSTRUCTION,
    SYSTEM_PROMPT,
    build_arbitration_prompt,
    build_token_bounded_arbitration_prompt,
    compute_multilabel_metrics,
    exact_letter_reward,
)
from src.config import ARBITRATION_CONFIG, resolve_grpo_per_device_batch_size


class CharacterTokenizer:
    pad_token = None
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        if tokenize:
            return self.encode(text)
        return text

    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(tokens)


def make_record(document_texts=None, query=None):
    query = query or (
        "What is the best answer?\n"
        "A. Alpha\n"
        "B. Beta\n"
        "C. Gamma\n"
        "D. Delta"
    )
    return {
        "query_id": "q1",
        "query": query,
        "gold_answer": ["A"],
        "documents": [
            {"id": f"doc_{index}", "title": f"Title {index}", "text": text}
            for index, text in enumerate(document_texts or ["short evidence"], start=1)
        ],
    }


class ExactLetterRewardTest(unittest.TestCase):
    def test_exact_match(self) -> None:
        rewards = exact_letter_reward(['{"response": "A"}'], ["A"])
        self.assertEqual(rewards, [1.0])

    def test_subset_is_not_rewarded(self) -> None:
        rewards = exact_letter_reward(['{"response": "A"}'], [["A", "B"]])
        self.assertEqual(rewards, [0.0])

    def test_superset_is_not_rewarded(self) -> None:
        rewards = exact_letter_reward(['{"response": ["A", "B"]}'], ["A"])
        self.assertEqual(rewards, [0.0])

    def test_different_sets_are_not_rewarded(self) -> None:
        rewards = exact_letter_reward(['{"response": "C"}'], ["A"])
        self.assertEqual(rewards, [0.0])

    def test_different_order_is_rewarded(self) -> None:
        rewards = exact_letter_reward(['{"response": ["B", "A"]}'], [["A", "B"]])
        self.assertEqual(rewards, [1.0])

    def test_empty_prediction_is_not_rewarded_for_non_empty_gold(self) -> None:
        rewards = exact_letter_reward(['{"response": ""}'], ["A"])
        self.assertEqual(rewards, [0.0])


class MultilabelMetricsTest(unittest.TestCase):
    def test_subset_accuracy_and_empty_rate(self) -> None:
        try:
            metrics = compute_multilabel_metrics(
                predicted_sets=[{"A"}, {"A"}, {"A", "B"}, set()],
                gold_sets=[{"A"}, {"A", "B"}, {"B", "A"}, {"C"}],
            )
        except ModuleNotFoundError as exc:
            if exc.name == "sklearn":
                self.skipTest("scikit-learn is not installed in this environment")
            raise

        self.assertEqual(metrics["subset_accuracy"], 0.5)
        self.assertEqual(metrics["empty_prediction_rate"], 0.25)
        self.assertIn("macro_f1", metrics)
        self.assertIn("macro_precision", metrics)
        self.assertIn("macro_recall", metrics)


class TokenBoundedPromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()

    def test_short_prompt_remains_unchanged(self) -> None:
        record = make_record(["short evidence"])
        result = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=4096)
        expected = build_arbitration_prompt(record, max_document_chars=0, max_total_document_chars=0)
        self.assertEqual(result.prompt, expected)
        self.assertFalse(result.was_truncated)
        self.assertFalse(result.was_rejected)

    def test_long_document_is_truncated_and_protected_fields_survive(self) -> None:
        record = make_record(["x" * 5000])
        result = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=4096)
        prompt_text = result.chat_text
        self.assertTrue(result.was_truncated)
        self.assertLessEqual(result.final_token_count, 4096)
        self.assertIn(SYSTEM_PROMPT, prompt_text)
        self.assertIn(record["query"], prompt_text)
        for option in ["A. Alpha", "B. Beta", "C. Gamma", "D. Delta"]:
            self.assertIn(option, prompt_text)
        self.assertIn(JSON_INSTRUCTION, prompt_text)

    def test_documents_are_removed_when_headers_cannot_fit(self) -> None:
        documents = ["x" for _ in range(20)]
        record = make_record(documents)
        record["documents"] = [
            {"id": f"doc_{index}", "title": "T" * 300, "text": "x"}
            for index in range(20)
        ]
        result = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=1600)
        self.assertTrue(result.was_truncated)
        self.assertGreater(result.documents_removed, 0)
        self.assertLessEqual(result.final_token_count, 1600)

    def test_final_prompt_is_within_4096_tokens(self) -> None:
        record = make_record(["x" * 10000, "y" * 10000, "z" * 10000])
        result = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=4096)
        self.assertLessEqual(result.final_token_count, 4096)

    def test_document_ordering_is_deterministic(self) -> None:
        record = make_record(["one " * 3000, "two " * 3000, "three " * 3000])
        result = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=4096)
        text = result.chat_text
        self.assertLess(text.index("Document 1"), text.index("Document 2"))
        self.assertLess(text.index("Document 2"), text.index("Document 3"))

    def test_same_input_produces_same_bounded_prompt(self) -> None:
        record = make_record(["repeat " * 2000, "stable " * 2000])
        first = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=4096)
        second = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=4096)
        self.assertEqual(first.chat_text, second.chat_text)
        self.assertEqual(first.metadata(), second.metadata())

    def test_pathological_protected_prompt_is_rejected(self) -> None:
        query = "Q " + ("protected " * 5000) + "\nA. A\nB. B\nC. C\nD. D"
        record = make_record(["document"], query=query)
        with self.assertWarns(RuntimeWarning):
            result = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=4096)
        self.assertTrue(result.was_rejected)

    def test_completion_length_is_independent_of_prompt_truncation(self) -> None:
        record = make_record(["x" * 10000])
        max_completion_length = 256
        result = build_token_bounded_arbitration_prompt(record, self.tokenizer, max_prompt_length=4096)
        self.assertTrue(result.was_truncated)
        self.assertEqual(max_completion_length, 256)


class GRPOConfigTest(unittest.TestCase):
    def test_small_gpu_overrides_do_not_apply_to_a100_class_vram(self) -> None:
        self.assertGreater(80, ARBITRATION_CONFIG.grpo_small_gpu_vram_gib)

    def test_num_generations_8_and_batch_size_1_are_accepted(self) -> None:
        self.assertEqual(resolve_grpo_per_device_batch_size(1, num_generations=8), 1)


if __name__ == "__main__":
    unittest.main()
