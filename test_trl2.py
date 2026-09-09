import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

print(hasattr(SFTConfig, "chunked_lm_head"))
