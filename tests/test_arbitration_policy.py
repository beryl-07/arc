from __future__ import annotations

import unittest

from src.arbitration_policy import compute_multilabel_metrics, exact_letter_reward


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


if __name__ == "__main__":
    unittest.main()
