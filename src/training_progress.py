"""Progress helpers for long training jobs."""

from __future__ import annotations

import time
from datetime import timedelta

from transformers import TrainerCallback


def _format_seconds(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


class ProgressTimeCallback(TrainerCallback):
    """Affiche une estimation simple du temps restant pendant l'entraînement."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.started_at: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started_at = time.time()
        if state.is_local_process_zero:
            total = state.max_steps if state.max_steps and state.max_steps > 0 else "unknown"
            print(f"[{self.label}] training started; total_steps={total}", flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.started_at is None or not state.is_local_process_zero:
            return
        elapsed = time.time() - self.started_at
        step = max(0, state.global_step)
        total = state.max_steps if state.max_steps and state.max_steps > 0 else None
        if total and step > 0:
            rate = elapsed / step
            eta = rate * max(0, total - step)
            progress = f"{step}/{total} ({step / total * 100:.1f}%)"
            timing = f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
        else:
            progress = str(step)
            timing = f"elapsed={_format_seconds(elapsed)} eta=unknown"
        metrics = ""
        if logs:
            visible = {
                key: value
                for key, value in logs.items()
                if key in {"loss", "eval_loss", "reward", "reward_std", "kl", "learning_rate"}
                or key.startswith("rewards/")
                or key.startswith("completions/")
            }
            if visible:
                metrics = " " + " ".join(f"{key}={value}" for key, value in visible.items())
        print(f"[{self.label}] step={progress} {timing}{metrics}", flush=True)
