"""
LoRA fine‑tuning with PEFT + bitsandbytes (QLoRA style).
Assumes you already ran prepare_faq.py → faq.jsonl
"""
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import bitsandbytes as bnb


MODEL_NAME = os.getenv("BASE_MODEL", "microsoft/phi-2")  # use smaller base if RAM limited
DATA_PATH = os.getenv("DATA_PATH", "faq.jsonl")
OUTPUT = os.getenv("OUTPUT_DIR", "lora-faq")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    ids = tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=1024,
    )
    ids["labels"] = ids["input_ids"].copy()
    return ids

print("Loading dataset…")
ds = load_dataset("json", data_files=DATA_PATH)["train"].map(tokenize, num_proc=4)

print("Loading base model…")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    device_map="auto",
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # adjust per model
)

model = get_peft_model(base_model, peft_cfg)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUTPUT,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    fp16=True,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()

model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print("✅ LoRA saved to", OUTPUT)
