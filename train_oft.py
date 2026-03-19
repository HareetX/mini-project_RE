import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import BOFTConfig, get_peft_model
from trl import SFTTrainer
import random
from huggingface_hub import snapshot_download

from src.duie_dataset import read_dataset, format_example_wo_schema, format_example_w_schema

import src.configs as configs


random.seed(42)  # Set random seed for reproducibility

# Download model
model_id = configs.MODEL_ID
local_dir = configs.LOCAL_DIR
snapshot_download(model_id, local_dir=local_dir)
model_id = local_dir

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Configure BOFT
boft_config = BOFTConfig(
    boft_block_size=configs.BOFT_BLOCK_SIZE,
    boft_n_butterfly_factor=configs.BOFT_N_BUTTERFLY_FACTOR,
    target_modules=configs.BOFT_TARGET_MODULES,
    boft_dropout=configs.BOFT_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM"
)

# Set up PEFT model with BOFT configuration
model = get_peft_model(model, boft_config)
model.print_trainable_parameters()

# Prepare training data
train_data = read_dataset(configs.TRAIN_PATH)
train_data = random.sample(train_data, 10000)  # Sample 10,000 examples for training

# Format examples into the required prompt structure
def format_prompts(example):
    return {"messages": format_example_wo_schema(example)}

dataset = Dataset.from_list(train_data)
dataset = dataset.map(format_prompts)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=configs.LOG_CHECKPOINTS_DIR,
    per_device_train_batch_size=configs.BATCH_SIZE,
    gradient_accumulation_steps=configs.GRADIENT_ACCUMULATION_STEPS,
    learning_rate=configs.LEARNING_RATE,
    logging_steps=1,
    # max_steps=100,              # Set 100 steps for quick testing
    num_train_epochs=configs.NUM_TRAIN_EPOCHS,
    save_strategy="steps",
    save_steps=10,
    optim="paged_adamw_32bit",
    fp16=False,
    bf16=True,
    report_to="tensorboard"
)

# Initialize the SFT trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

# Start training
print("Start training...")
trainer.train()

# Save the fine-tuned model
save_dir = configs.FINETUNED_MODEL_DIR
trainer.model.save_pretrained(save_dir)
print(f"Saved fine-tuned model to {save_dir}")
