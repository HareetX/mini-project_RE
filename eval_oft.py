import json
import os
from typing import List, Dict
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.duie_dataset import read_dataset, format_example_wo_schema
from utils import parse_json


def test_oft(batch_size: int = 8) -> List[Dict]:
	random.seed(42)  # 设置随机种子以确保结果可复现
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Paths
	base_model_dir = os.path.join("models", "qwen2.5-1.5b-instruct")
	finetuned_dir = "./qwen-oft-final"
	dev_path = "data/dev.json"
	save_path = "eval_oft_results.jsonl"

	# Check if eval results already exist
	if os.path.exists(save_path):
		tqdm.write(f"Evaluation results already exist at {save_path}. Loading...")
		with open(save_path, "r", encoding="utf-8") as f:
			results = [json.loads(line) for line in f]
		tqdm.write(f"Loaded {len(results)} evaluation results.")
		return results

	# Load tokenizer once
	tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "left"  # 确保左侧填充以适应因果语言模型

	# Load model and adapter
	model = AutoModelForCausalLM.from_pretrained(
		base_model_dir,
		device_map="auto",
		torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
		trust_remote_code=True,
	)

	model = PeftModel.from_pretrained(model, finetuned_dir)
	model.eval()

	# Load eval data
	dev_data = read_dataset(dev_path)
	dev_data = random.sample(dev_data, 10)  # 仅使用 200 条数据进行快速测试，实际评估时请使用全部数据

	# Get all messages
	all_messages = [format_example_wo_schema(example, is_query=True) for example in dev_data]

	def batch_generate(messages_batch):
		# Generate predictions for a batch of messages
		texts = [tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
		inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

		with torch.no_grad():
			outputs = model.generate(
				**inputs,
				max_new_tokens=512,
				pad_token_id=tokenizer.pad_token_id,
				do_sample=False,
			)

		input_length = inputs.input_ids.shape[1]
		generated_tokens = outputs[:, input_length:]
		responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

		return responses

	base_preds = []
	ft_preds = []

	# Base model predictions (without adapter)
	tqdm.write("Generating predictions with base model...")
	with model.disable_adapter():
		for i in tqdm(range(0, len(all_messages), batch_size), desc="Base Model"):
			batch = all_messages[i:i+batch_size]
			raw_preds = batch_generate(batch)
			base_preds.extend(raw_preds)

	# Finetuned model predictions (with adapter)
	tqdm.write("Generating predictions with finetuned model...")
	for i in tqdm(range(0, len(all_messages), batch_size), desc="Finetuned Model"):
		batch = all_messages[i:i+batch_size]
		raw_preds = batch_generate(batch)
		ft_preds.extend(raw_preds)

	# Parse and save results
	results = []
	for idx, (example, base_pred, ft_pred) in enumerate(zip(dev_data, base_preds, ft_preds), 1):
		results.append({
			"id": idx,
			"text": example.get("text", ""),
			"ground_truth": example.get("spo_list", []),
			"base_pred": base_pred,
			"finetuned_pred": ft_pred,
		})

	# Save JSONL
	with open(save_path, "w", encoding="utf-8") as f:
		for row in results:
			f.write(json.dumps(row, ensure_ascii=False) + "\n")

	tqdm.write(f"Done. Saved results to {save_path}")

	return results


def eval_oft():
	results = test_oft()

	# Accuracy evaluation (exact match)
	correct_base = 0
	correct_ft = 0
	for res in results:
		# Convert ground truth triplets to a set of strings for easy comparison
		gt = res["ground_truth"]
		gt_strs = set()
		for gt_triplet in gt:
			sub = gt_triplet.get("subject", "")
			obj = gt_triplet.get("object", {}).get("@value", "")
			predicate = gt_triplet.get("predicate", "")
			gt_str = f"{sub} {predicate} {obj}"
			gt_strs.add(gt_str)

		# Base model prediction
		base_preds = res["base_pred"]
		base_preds_json = parse_json(base_preds)
		base_pred_strs = set()
		if base_preds_json is not None:
			base_preds_json = base_preds_json.get("extracted_triplets", [])
			try:
				for pred in base_preds_json:
					sub = pred.get("subject", "")
					obj = pred.get("object", "")
					predicate = pred.get("predicate", "")
					pred_str = f"{sub} {predicate} {obj}"
					base_pred_strs.add(pred_str)
			except Exception as e:
				print(f"Error parsing base model prediction for example {res['id']}: {e}")
				print(f"Base prediction string: {base_preds}")

		# Finetuned model prediction
		ft_preds = res["finetuned_pred"]
		ft_preds_json = parse_json(ft_preds)
		ft_pred_strs = set()
		if ft_preds_json is not None:
			ft_preds_json = ft_preds_json.get("extracted_triplets", [])
			try:
				for pred in ft_preds_json:
					sub = pred.get("subject", "")
					obj = pred.get("object", "")
					predicate = pred.get("predicate", "")
					pred_str = f"{sub} {predicate} {obj}"
					ft_pred_strs.add(pred_str)
					print(f"Example {res['id']} - Finetuned prediction: {pred_str}")
			except Exception as e:
				print(f"Error parsing finetuned model prediction for example {res['id']}: {e}")
				print(f"Finetuned prediction string: {ft_preds}")

		# Calculate token accuracy
		correct_base_tmp = 0
		base_counter = 0
		correct_ft_tmp = 0
		ft_counter = 0
		for gt_str in gt_strs:
			gt_tokens = set(gt_str.split())
			for base_pred_str in base_pred_strs:
				base_tokens = set(base_pred_str.split())
				correct_base_tmp += len(gt_tokens & base_tokens) / len(gt_tokens) if len(gt_tokens) > 0 else 0
				base_counter += 1

			for ft_pred_str in ft_pred_strs:
				ft_tokens = set(ft_pred_str.split())
				correct_ft_tmp += len(gt_tokens & ft_tokens) / len(gt_tokens) if len(gt_tokens) > 0 else 0
				ft_counter += 1
		correct_base += correct_base_tmp / base_counter if base_counter > 0 else 0
		correct_ft += correct_ft_tmp / ft_counter if ft_counter > 0 else 0

	total = len(results)
	base_acc = correct_base / total if total > 0 else 0
	ft_acc = correct_ft / total if total > 0 else 0
	print(f"Base Model Accuracy: {base_acc:.4f} ({correct_base}/{total})")
	print(f"Finetuned Model Accuracy: {ft_acc:.4f} ({correct_ft	}/{total})")

if __name__ == "__main__":
	eval_oft()
