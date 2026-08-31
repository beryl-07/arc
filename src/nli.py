"""NLI model wrapper used by conflict detectors."""

from __future__ import annotations

from .config import PipelineConfig


class NLIClassifier:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "text-classification",
                model=self.config.nli_model,
                device=device,
                truncation=True,
                max_length=512,
            )
            print(f"NLI pipeline loaded ({self.config.nli_model}).")
        return self._pipeline

    def classify_pairs(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Classify text pairs in batches.

        Hugging Face NLI pipelines accept a list of {"text", "text_pair"} inputs.
        Some sequence-classification checkpoints ignore text_pair in older
        pipeline versions, so we fail loudly instead of silently scoring only the
        first text.
        """

        outputs: list[dict] = []
        for start in range(0, len(pairs), self.config.nli_batch_size):
            batch_pairs = pairs[start : start + self.config.nli_batch_size]
            batch = [
                {"text": left[:512], "text_pair": right[:512]}
                for left, right in batch_pairs
            ]
            outputs.extend(self.pipeline(batch))
        return outputs

    def is_contradiction(self, result: dict) -> bool:
        return (
            result["label"].lower() == self.config.nli_contradiction
            and float(result["score"]) >= self.config.nli_threshold
        )
