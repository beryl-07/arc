"""Project configuration for the Phase 1 conflict-construction pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv()


def _csv_env(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime settings with Colab-compatible defaults.

    Every value can be overridden by environment variables, which keeps paths and
    private tokens out of source files.
    """

    java_options: str = os.getenv("_JAVA_OPTIONS", "-Xmx10g")
    hf_home: str = os.getenv("HF_HOME", "/content/drive/MyDrive/hf_cache")
    hf_token: str = os.getenv("HF_TOKEN", "")
    openai_api_key: str = os.getenv(
        "OPENAI_API_KEY", "sk-unused-arc-placeholder"
    )

    corpus_name: str = os.getenv("CORPUS_NAME", "MedText")

    top_k: int = int(os.getenv("TOP_K", "3"))
    retrievers: list[str] = field(
        default_factory=lambda: _csv_env(
            "RETRIEVERS", ["BM25", "Contriever", "SPECTER", "MedCPT"]
        )
    )

    use_drive: bool = _bool_env("USE_DRIVE", True)
    drive_root: Path = Path(os.getenv("DRIVE_ROOT", "/content/drive/MyDrive/arc_phase1"))
    local_root: Path = Path(os.getenv("LOCAL_ROOT", "/content/arc_phase1"))
    local_embed_dir: Path = Path(os.getenv("LOCAL_EMBED_DIR", "/content/pubmed_embeddings"))

    nli_model: str = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    nli_contradiction: str = os.getenv("NLI_CONTRADICTION", "contradiction")
    nli_threshold: float = float(os.getenv("NLI_THRESHOLD", "0.7"))
    nli_batch_size: int = int(os.getenv("NLI_BATCH_SIZE", "32"))

    pke_model: str = os.getenv("PKE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    n_probes: int = int(os.getenv("N_PROBES", "7"))
    pke_temperature: float = float(os.getenv("PKE_TEMPERATURE", "0.8"))
    pke_known_threshold: float = float(os.getenv("PKE_KNOWN_THRESH", "0.6"))
    pke_max_new_tokens: int = int(os.getenv("PKE_MAX_NEW_TOKENS", "256"))

    default_n_questions: int = int(os.getenv("DEFAULT_N_QUESTIONS", "10"))

    @property
    def resolved_corpus_name(self) -> str:
        return self.corpus_name

    @property
    def root_dir(self) -> Path:
        return self.drive_root if self.use_drive else self.local_root

    @property
    def db_dir(self) -> Path:
        return self.root_dir / "corpus"

    @property
    def medrag_repo_dir(self) -> Path:
        return self.root_dir / "MedRAG"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "outputs"

    def as_metadata(self) -> dict:
        return {
            "top_k": self.top_k,
            "corpus_name": self.resolved_corpus_name,
            "retrievers": self.retrievers,
            "nli_model": self.nli_model,
            "nli_threshold": self.nli_threshold,
            "pke_model": self.pke_model,
            "n_probes": self.n_probes,
        }


def load_config() -> PipelineConfig:
    config = PipelineConfig()
    os.environ["_JAVA_OPTIONS"] = config.java_options
    os.environ.setdefault("OPENAI_API_KEY", config.openai_api_key)
    os.environ["HF_HOME"] = config.hf_home
    if config.hf_token:
        os.environ["HF_TOKEN"] = config.hf_token
    return config


CONFIG = load_config()

# Backward-compatible dictionary for early notebook-style imports.
config = {
    "_JAVA_OPTIONS": CONFIG.java_options,
    "OPENAI_API_KEY": CONFIG.openai_api_key,
    "HF_HOME": CONFIG.hf_home,
    "HF_TOKEN": CONFIG.hf_token,
    "CORPUS_NAME": CONFIG.resolved_corpus_name,
    "TOP_K": CONFIG.top_k,
    "RETRIEVERS": CONFIG.retrievers,
    "USE_DRIVE": CONFIG.use_drive,
    "ROOT_DIR": str(CONFIG.root_dir),
    "DB_DIR": str(CONFIG.db_dir),
    "MEDRAG_REPO_DIR": str(CONFIG.medrag_repo_dir),
    "OUTPUT_DIR": str(CONFIG.output_dir),
    "LOCAL_EMBED_DIR": str(CONFIG.local_embed_dir),
    "NLI_MODEL": CONFIG.nli_model,
    "NLI_CONTRADICTION": CONFIG.nli_contradiction,
    "NLI_THRESHOLD": CONFIG.nli_threshold,
    "PKE_MODEL": CONFIG.pke_model,
    "N_PROBES": CONFIG.n_probes,
    "PKE_TEMPERATURE": CONFIG.pke_temperature,
    "PKE_KNOWN_THRESH": CONFIG.pke_known_threshold,
    "DEFAULT_N_QUESTIONS": CONFIG.default_n_questions,
}
