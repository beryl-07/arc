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


def _optional_path_env(name: str, default: str | None = None) -> Path | None:
    value = os.getenv(name, default)
    return Path(value) if value else None


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


@dataclass(frozen=True)
class GRPOHardwareProfile:
    """Named GRPO profile used to make hardware assumptions explicit."""

    name: str
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    beta: float
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    optimizer: str


A100_GRPO_PROFILE = GRPOHardwareProfile(
    name="a100_80gb_single_gpu",
    num_generations=8,
    max_prompt_length=4096,
    max_completion_length=256,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    beta=0.01,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True,
    optimizer="paged_adamw_8bit",
)


@dataclass(frozen=True)
class ArbitrationTrainingConfig:
    """Configuration centrale du pipeline SFT / GRPO / évaluation."""

    # =========================
    # Configuration SFT
    # =========================
    sft_num_epochs: int = int(os.getenv("SFT_NUM_EPOCHS", "3"))
    sft_learning_rate: float = float(os.getenv("SFT_LEARNING_RATE", "2e-5"))
    sft_max_seq_length: int = int(os.getenv("SFT_MAX_SEQ_LENGTH", "2048"))
    sft_save_steps: int = int(os.getenv("SFT_SAVE_STEPS", "100"))
    # Obligatoire au lancement SFT : la valeur par défaut documente l'absence
    # volontaire de validation implicite.
    sft_val_path: Path | None = _optional_path_env("SFT_VAL_PATH", None)
    sft_per_device_train_batch_size: int = int(os.getenv("SFT_PER_DEVICE_TRAIN_BATCH_SIZE", "4"))
    sft_gradient_accumulation_steps: int = int(os.getenv("SFT_GRADIENT_ACCUMULATION_STEPS", "4"))
    sft_logging_steps: int = int(os.getenv("SFT_LOGGING_STEPS", "10"))
    lora_r: int = int(os.getenv("LORA_R", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.05"))
    lora_target_modules: list[str] = field(
        default_factory=lambda: _csv_env("LORA_TARGET_MODULES", ["q_proj", "k_proj", "v_proj", "o_proj"])
    )

    # =========================
    # Configuration GRPO
    # =========================
    grpo_profile: GRPOHardwareProfile = A100_GRPO_PROFILE
    grpo_num_generations: int = int(os.getenv("GRPO_NUM_GENERATIONS", str(A100_GRPO_PROFILE.num_generations)))
    # 50 est un point de départ expérimental, pas une valeur optimale. Si le
    # coût calcul le permet, comparer ensuite avec 80 steps.
    grpo_max_steps: int = int(os.getenv("GRPO_MAX_STEPS", "50"))
    grpo_learning_rate: float = float(os.getenv("GRPO_LEARNING_RATE", str(A100_GRPO_PROFILE.learning_rate)))
    grpo_lr_scheduler_type: str = os.getenv("GRPO_LR_SCHEDULER_TYPE", "constant_with_warmup")
    grpo_warmup_ratio: float = float(os.getenv("GRPO_WARMUP_RATIO", "0.03"))
    grpo_beta: float = float(os.getenv("GRPO_BETA", str(A100_GRPO_PROFILE.beta)))
    grpo_max_prompt_length: int = int(os.getenv("GRPO_MAX_PROMPT_LENGTH", str(A100_GRPO_PROFILE.max_prompt_length)))
    grpo_max_completion_length: int = int(os.getenv("GRPO_MAX_COMPLETION_LENGTH", str(A100_GRPO_PROFILE.max_completion_length)))
    grpo_per_device_batch_size: int = int(
        os.getenv("GRPO_PER_DEVICE_BATCH_SIZE", str(A100_GRPO_PROFILE.per_device_train_batch_size))
    )
    grpo_gradient_accumulation_steps: int = int(
        os.getenv("GRPO_GRADIENT_ACCUMULATION_STEPS", str(A100_GRPO_PROFILE.gradient_accumulation_steps))
    )
    grpo_save_steps: int = int(os.getenv("GRPO_SAVE_STEPS", "50"))
    grpo_logging_steps: int = int(os.getenv("GRPO_LOGGING_STEPS", "5"))
    grpo_bf16: bool = _bool_env("GRPO_BF16", A100_GRPO_PROFILE.bf16)
    grpo_fp16: bool = _bool_env("GRPO_FP16", A100_GRPO_PROFILE.fp16)
    grpo_gradient_checkpointing: bool = _bool_env(
        "GRPO_GRADIENT_CHECKPOINTING", A100_GRPO_PROFILE.gradient_checkpointing
    )
    grpo_optimizer: str = os.getenv("GRPO_OPTIMIZER", A100_GRPO_PROFILE.optimizer)
    # Profil optionnel pour exécuter GRPO sur des GPU 16 Go type T4.
    grpo_small_gpu_vram_gib: float = float(os.getenv("GRPO_SMALL_GPU_VRAM_GIB", "24"))
    grpo_memory_safe_max_steps: int = int(os.getenv("GRPO_MEMORY_SAFE_MAX_STEPS", "200"))
    grpo_memory_safe_num_generations: int = int(os.getenv("GRPO_MEMORY_SAFE_NUM_GENERATIONS", "2"))
    grpo_memory_safe_max_prompt_length: int = int(os.getenv("GRPO_MEMORY_SAFE_MAX_PROMPT_LENGTH", "384"))
    grpo_memory_safe_max_completion_length: int = int(os.getenv("GRPO_MEMORY_SAFE_MAX_COMPLETION_LENGTH", "96"))
    grpo_memory_safe_optimizer: str = os.getenv("GRPO_MEMORY_SAFE_OPTIMIZER", "paged_adamw_8bit")

    # =========================
    # Évaluation
    # =========================
    eval_multilabel: bool = _bool_env("EVAL_MULTILABEL", True)

    model_name: str = os.getenv("ARBITRATION_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    sft_train_path: Path = Path(os.getenv("SFT_TRAIN_PATH", "data/sft_execution_data.jsonl"))
    grpo_train_path: Path = Path(os.getenv("GRPO_TRAIN_PATH", "data/grpo_train.jsonl"))
    val_path: Path = Path(os.getenv("ARBITRATION_VAL_PATH", "data/val.jsonl"))
    test_path: Path = Path(os.getenv("ARBITRATION_TEST_PATH", "data/test.jsonl"))
    sft_output_dir: Path = Path(os.getenv("SFT_OUTPUT_DIR", "models/sft_arbitration_policy"))
    grpo_output_dir: Path = Path(os.getenv("GRPO_OUTPUT_DIR", "models/grpo_arbitration_policy"))
    outputs_dir: Path = Path(os.getenv("ARBITRATION_OUTPUTS_DIR", "outputs"))
    eval_max_input_length: int = int(os.getenv("EVAL_MAX_INPUT_LENGTH", str(A100_GRPO_PROFILE.max_prompt_length)))
    eval_max_new_tokens: int = int(os.getenv("EVAL_MAX_NEW_TOKENS", "128"))
    max_document_chars: int = int(os.getenv("ARBITRATION_MAX_DOCUMENT_CHARS", "0"))
    max_total_document_chars: int = int(os.getenv("ARBITRATION_MAX_TOTAL_DOCUMENT_CHARS", "0"))
    seed: int = int(os.getenv("ARBITRATION_SEED", "42"))


ARBITRATION_CONFIG = ArbitrationTrainingConfig()


def resolve_grpo_per_device_batch_size(
    num_processes: int,
    config: ArbitrationTrainingConfig = ARBITRATION_CONFIG,
    num_generations: int | None = None,
) -> int:
    """Retourne un batch GRPO compatible avec num_generations pour Accelerate."""

    if num_processes <= 0:
        raise ValueError("num_processes must be positive")

    generations = num_generations or config.grpo_num_generations
    batch_size = config.grpo_per_device_batch_size
    if batch_size == 1:
        return batch_size
    while (num_processes * batch_size) % generations != 0:
        batch_size += 1
    return batch_size


# Aliases explicites demandés par le cahier des charges. Ils pointent tous vers
# la configuration centrale ci-dessus pour éviter les divergences.
SFT_NUM_EPOCHS = ARBITRATION_CONFIG.sft_num_epochs
SFT_LEARNING_RATE = ARBITRATION_CONFIG.sft_learning_rate
SFT_MAX_SEQ_LENGTH = ARBITRATION_CONFIG.sft_max_seq_length
SFT_SAVE_STEPS = ARBITRATION_CONFIG.sft_save_steps
SFT_VAL_PATH = ARBITRATION_CONFIG.sft_val_path
GRPO_NUM_GENERATIONS = ARBITRATION_CONFIG.grpo_num_generations
GRPO_MAX_STEPS = ARBITRATION_CONFIG.grpo_max_steps
GRPO_LEARNING_RATE = ARBITRATION_CONFIG.grpo_learning_rate
GRPO_LR_SCHEDULER_TYPE = ARBITRATION_CONFIG.grpo_lr_scheduler_type
GRPO_WARMUP_RATIO = ARBITRATION_CONFIG.grpo_warmup_ratio
GRPO_BETA = ARBITRATION_CONFIG.grpo_beta
GRPO_MAX_PROMPT_LENGTH = ARBITRATION_CONFIG.grpo_max_prompt_length
GRPO_MAX_COMPLETION_LENGTH = ARBITRATION_CONFIG.grpo_max_completion_length
GRPO_PER_DEVICE_BATCH_SIZE = ARBITRATION_CONFIG.grpo_per_device_batch_size
GRPO_GRADIENT_ACCUMULATION_STEPS = ARBITRATION_CONFIG.grpo_gradient_accumulation_steps
GRPO_BF16 = ARBITRATION_CONFIG.grpo_bf16
GRPO_FP16 = ARBITRATION_CONFIG.grpo_fp16
GRPO_GRADIENT_CHECKPOINTING = ARBITRATION_CONFIG.grpo_gradient_checkpointing
GRPO_OPTIMIZER = ARBITRATION_CONFIG.grpo_optimizer
GRPO_SMALL_GPU_VRAM_GIB = ARBITRATION_CONFIG.grpo_small_gpu_vram_gib
GRPO_MEMORY_SAFE_MAX_STEPS = ARBITRATION_CONFIG.grpo_memory_safe_max_steps
GRPO_MEMORY_SAFE_NUM_GENERATIONS = ARBITRATION_CONFIG.grpo_memory_safe_num_generations
GRPO_MEMORY_SAFE_MAX_PROMPT_LENGTH = ARBITRATION_CONFIG.grpo_memory_safe_max_prompt_length
GRPO_MEMORY_SAFE_MAX_COMPLETION_LENGTH = ARBITRATION_CONFIG.grpo_memory_safe_max_completion_length
EVAL_MULTILABEL = ARBITRATION_CONFIG.eval_multilabel

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
    "SFT_NUM_EPOCHS": SFT_NUM_EPOCHS,
    "SFT_LEARNING_RATE": SFT_LEARNING_RATE,
    "SFT_MAX_SEQ_LENGTH": SFT_MAX_SEQ_LENGTH,
    "SFT_SAVE_STEPS": SFT_SAVE_STEPS,
    "SFT_VAL_PATH": str(SFT_VAL_PATH) if SFT_VAL_PATH else None,
    "GRPO_NUM_GENERATIONS": GRPO_NUM_GENERATIONS,
    "GRPO_MAX_STEPS": GRPO_MAX_STEPS,
    "GRPO_LEARNING_RATE": GRPO_LEARNING_RATE,
    "GRPO_LR_SCHEDULER_TYPE": GRPO_LR_SCHEDULER_TYPE,
    "GRPO_WARMUP_RATIO": GRPO_WARMUP_RATIO,
    "GRPO_BETA": GRPO_BETA,
    "GRPO_MAX_PROMPT_LENGTH": GRPO_MAX_PROMPT_LENGTH,
    "GRPO_MAX_COMPLETION_LENGTH": GRPO_MAX_COMPLETION_LENGTH,
    "GRPO_PER_DEVICE_BATCH_SIZE": GRPO_PER_DEVICE_BATCH_SIZE,
    "GRPO_GRADIENT_ACCUMULATION_STEPS": GRPO_GRADIENT_ACCUMULATION_STEPS,
    "GRPO_BF16": GRPO_BF16,
    "GRPO_FP16": GRPO_FP16,
    "GRPO_GRADIENT_CHECKPOINTING": GRPO_GRADIENT_CHECKPOINTING,
    "GRPO_OPTIMIZER": GRPO_OPTIMIZER,
    "GRPO_SMALL_GPU_VRAM_GIB": GRPO_SMALL_GPU_VRAM_GIB,
    "GRPO_MEMORY_SAFE_MAX_STEPS": GRPO_MEMORY_SAFE_MAX_STEPS,
    "GRPO_MEMORY_SAFE_NUM_GENERATIONS": GRPO_MEMORY_SAFE_NUM_GENERATIONS,
    "GRPO_MEMORY_SAFE_MAX_PROMPT_LENGTH": GRPO_MEMORY_SAFE_MAX_PROMPT_LENGTH,
    "GRPO_MEMORY_SAFE_MAX_COMPLETION_LENGTH": GRPO_MEMORY_SAFE_MAX_COMPLETION_LENGTH,
    "EVAL_MULTILABEL": EVAL_MULTILABEL,
}
