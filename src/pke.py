"""Parametric Knowledge Estimation with repeated LLM probes."""

from __future__ import annotations

import logging

from .config import PipelineConfig


LOG = logging.getLogger(__name__)


class ParametricKnowledgeEstimator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._model = None
        self._tokenizer = None

    def load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            LOG.info("Loading PKE LLM: model=%s", self.config.pke_model)
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.pke_model)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.pke_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            LOG.info("PKE LLM loaded: model=%s", self.config.pke_model)
        return self._model, self._tokenizer

    def generate(self, question: str) -> str:
        import torch

        model, tokenizer = self.load_model()
        prompt = (
            "You are a medical expert. Answer the following biomedical question "
            f"concisely and factually.\n\nQuestion: {question}\n\nAnswer:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.config.pke_max_new_tokens,
                temperature=self.config.pke_temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def estimate(self, question: str) -> dict:
        probes = []
        LOG.info("Running PKE probes: n_probes=%s", self.config.n_probes)
        for idx in range(self.config.n_probes):
            answer = self.generate(question)
            probes.append(answer)
            LOG.info("PKE probe finished: index=%s/%s preview=%r", idx + 1, self.config.n_probes, answer[:80])

        return {"probes": probes}
