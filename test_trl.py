import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

data = [
    {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]},
    {"messages": [{"role": "user", "content": "world"}, {"role": "assistant", "content": "earth"}]},
]
dataset = Dataset.from_list(data)

def format_chat(example):
    print("TYPE OF EXAMPLE:", type(example))
    print("KEYS:", example.keys())
    print("MESSAGES TYPE:", type(example["messages"]))
    if len(example["messages"]) > 0:
        print("FIRST MESSAGE TYPE:", type(example["messages"][0]))
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    # returning a list or string?
    # Let's try returning a list of strings, which is what old TRL wanted
    return text

try:
    trainer = SFTTrainer(
        model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_chat,
        dataset_text_field="text",
        max_seq_length=128,
        packing=False
    )
    print("SUCCESS with returning string")
except Exception as e:
    print("FAILED with string:", type(e), e)

def format_chat_list(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return [text]

try:
    trainer = SFTTrainer(
        model=None,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_chat_list,
        dataset_text_field="text",
        max_seq_length=128,
        packing=False
    )
    print("SUCCESS with returning list of string")
except Exception as e:
    print("FAILED with list of string:", type(e), e)
