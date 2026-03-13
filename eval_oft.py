import json
import os
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.duie_dataset import read_dataset, format_example_wo_schema


def build_input_ids(tokenizer, messages):
	"""Convert chat messages to model input IDs."""
	encoded = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		return_tensors="pt",
	)

	return encoded


@torch.inference_mode()
def generate_triplets(model, tokenizer, messages: List[Dict], device: torch.device, max_new_tokens=128) -> str:
    inputs = build_input_ids(tokenizer, messages).to(device)
    output_ids = model.generate(
		inputs.input_ids,
		max_new_tokens=max_new_tokens,
		do_sample=False,
		pad_token_id=tokenizer.eos_token_id,
        attention_mask=inputs.attention_mask,
	)
	# Strip the prompt part
    gen_ids = output_ids[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Paths
	base_model_dir = os.path.join("models", "qwen2.5-1.5b-instruct")
	finetuned_dir = "./qwen-oft-final"
	dev_path = "data/dev.json"
	save_path = "eval_oft_results.jsonl"

	# Load tokenizer once
	tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
	tokenizer.pad_token = tokenizer.eos_token

	# Base model
	base_model = AutoModelForCausalLM.from_pretrained(
		base_model_dir,
		device_map="auto",
		torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
		trust_remote_code=True,
	)

	# Finetuned model (base + adapter)
	ft_model = AutoModelForCausalLM.from_pretrained(
		base_model_dir,
		device_map="auto",
		torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
		trust_remote_code=True,
	)
	ft_model = PeftModel.from_pretrained(ft_model, finetuned_dir)

	# Load eval data
	dev_data = read_dataset(dev_path)

	results = []
	for idx, example in enumerate(dev_data, 1):
		messages = format_example_wo_schema(example)
		base_pred = generate_triplets(base_model, tokenizer, messages, device)
		ft_pred = generate_triplets(ft_model, tokenizer, messages, device)
		results.append({
			"id": idx,
			"text": example.get("text", ""),
			"ground_truth": example.get("spo_list", []),
			"base_pred": base_pred,
			"finetuned_pred": ft_pred,
		})

		if idx % 20 == 0:
			print(f"Processed {idx}/{len(dev_data)} examples")

	# Save JSONL
	with open(save_path, "w", encoding="utf-8") as f:
		for row in results:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")

	print(f"Done. Saved results to {save_path}")


if __name__ == "__main__":
	main()
