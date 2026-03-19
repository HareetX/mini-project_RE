import difflib
import json
import os
from typing import List, Dict, Set
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

from src.duie_dataset import read_dataset, format_example_wo_schema
from src.utils import parse_json

import src.configs as configs


def test_oft(batch_size: int = 8) -> List[Dict]:
	random.seed(42)  # Set random seed
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# Paths
	base_model_dir = configs.LOCAL_DIR
	finetuned_dir = configs.FINETUNED_MODEL_DIR
	dev_path = configs.VALID_PATH
	save_path = configs.EVALUATION_RESULTS_PATH

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
	tokenizer.padding_side = "left"

	# Load model and adapter
	model = AutoModelForCausalLM.from_pretrained(
		base_model_dir,
		device_map="auto",
		torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
		trust_remote_code=True,
	)

	model = PeftModel.from_pretrained(model, finetuned_dir).to(device)
	model.eval()

	# Load eval data
	dev_data = read_dataset(dev_path)
	dev_data = random.sample(dev_data, 200)  # Sample 200 examples for evaluation

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

def calculate_re_metrics(ground_truths: List[Dict], preds: List[Dict], match_type: Set[str] = configs.EXACT_MATCH_TYPES) -> Dict[str, float]:
	gt_set = set()
	for gt in ground_truths:
		try:
			item = []
			if "subject" in match_type:
				item.append(gt.get("subject", ""))
			if "predicate" in match_type:
				item.append(gt.get("predicate", ""))
			if "object" in match_type:
				item.append(gt.get("object", {}).get("@value", ""))
			item = tuple(item)

			gt_set.add(item)
		except Exception as e:
			print(f"Error processing ground truth: {gt}, error: {e}")

	pred_set = set()
	for pred in preds:
		try:
			item = []
			if "subject" in match_type:
				item.append(pred.get("subject", ""))
			if "predicate" in match_type:
				item.append(pred.get("predicate", ""))
			if "object" in match_type:
				item.append(pred.get("object", ""))
			item = tuple(item)

			pred_set.add(item)
		except Exception as e:
			print(f"Error processing prediction: {pred}, error: {e}")

	tp = len(gt_set & pred_set)
	fp = len(pred_set - gt_set)
	fn = len(gt_set - pred_set)

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

	return {
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"tp": tp,
		"fp": fp,
		"fn": fn,
	}

def string_similarity(str1: str, str2: str) -> float:
	if not str1 or not str2:
		return 0.0
	return difflib.SequenceMatcher(None, str1, str2).ratio()

def calculate_re_metrics_soft_matching(ground_truths: List[Dict], preds: List[Dict], threshold: float = 0.66) -> Dict[str, float]:
	gt_list = [
		(
			gt.get("subject", ""),
			gt.get("predicate", ""),
			gt.get("object", {}).get("@value", ""),
		) for gt in ground_truths
	]
	pred_list = []
	for pred in preds:
		try:
			item = (
				pred.get("subject", ""),
				pred.get("predicate", ""),
				pred.get("object", ""),
			)
			pred_list.append(item)
		except Exception as e:
			print(f"Error processing prediction: {pred}, error: {e}")

	score_matrix = []
	for i, gt in enumerate(gt_list):
		for j, pred in enumerate(pred_list):
			sub_sim = string_similarity(gt[0], pred[0])
			pred_sim = string_similarity(gt[1], pred[1])
			obj_sim = string_similarity(gt[2], pred[2])

			avg_sim = (sub_sim + pred_sim + obj_sim) / 3
			score_matrix.append((i, j, avg_sim))

	# Greedy matching
	score_matrix.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity

	matched_gt = set()
	matched_pred = set()
	tp_count = 0

	for gt_idx, pred_idx, score in score_matrix:
		if score < threshold:
			break

		if gt_idx not in matched_gt and pred_idx not in matched_pred:
			matched_gt.add(gt_idx)
			matched_pred.add(pred_idx)
			tp_count += 1

	tp = tp_count
	fp = len(pred_list) - tp
	fn = len(gt_list) - tp

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

	return {
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"tp": tp,
		"fp": fp,
		"fn": fn,
	}

def calculate_re_metrics_semantic(ground_truths: List[Dict], preds: List[Dict], model: SentenceTransformer, threshold: float = 0.66) -> Dict[str, float]:
	gt_sentences = [
		f"{gt.get('subject', '')} {gt.get('predicate', '')} {gt.get('object', {}).get('@value', '')}"
		for gt in ground_truths
	]

	pred_sentences = []
	for pred in preds:
		try:
			sent = f"{pred.get('subject', '')} {pred.get('predicate', '')} {pred.get('object', '')}"
			pred_sentences.append(sent)
		except Exception as e:
			print(f"Error processing prediction: {pred}, error: {e}")

	if not pred_sentences:
		return {
			"precision": 0.0,
			"recall": 0.0,
			"f1": 0.0,
			"tp": 0,
			"fp": 0,
			"fn": len(gt_sentences),
		}

	gt_embeddings = model.encode(gt_sentences, convert_to_tensor=True)
	pred_embeddings = model.encode(pred_sentences, convert_to_tensor=True)

	cosine_scores = util.cos_sim(gt_embeddings, pred_embeddings).cpu().numpy()

	score_matrix = []
	for i in range(len(gt_sentences)):
		for j in range(len(pred_sentences)):
			score_matrix.append((i, j, cosine_scores[i][j]))

	# Greedy matching
	score_matrix.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity

	matched_gt = set()
	matched_pred = set()
	tp_count = 0

	for gt_idx, pred_idx, score in score_matrix:
		if score < threshold:
			break

		if gt_idx not in matched_gt and pred_idx not in matched_pred:
			matched_gt.add(gt_idx)
			matched_pred.add(pred_idx)
			tp_count += 1

	tp = tp_count
	fp = len(pred_sentences) - tp
	fn = len(gt_sentences) - tp

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

	return {
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"tp": tp,
		"fp": fp,
		"fn": fn,
	}

def eval_oft_exact_matching(results: List[Dict]) -> Dict[str, Dict[str, float]]:
	total = len(results)

	avg_metrics_base = {"precision": 0, "recall": 0, "f1": 0}
	avg_metrics_ft = {"precision": 0, "recall": 0, "f1": 0}

	for res in results:
		metrics_base = calculate_re_metrics(
      		res["ground_truth"],
        	parse_json(res["base_pred"]).get("extracted_triplets", []) if parse_json(res["base_pred"]) is not None else [],
        	match_type=configs.EXACT_MATCH_TYPES,
		)
		metrics_ft = calculate_re_metrics(
      		res["ground_truth"],
        	parse_json(res["finetuned_pred"]).get("extracted_triplets", []) if parse_json(res["finetuned_pred"]) is not None else [],
         	match_type=configs.EXACT_MATCH_TYPES,
        )

		avg_metrics_base["precision"] += metrics_base["precision"]
		avg_metrics_base["recall"] += metrics_base["recall"]
		avg_metrics_base["f1"] += metrics_base["f1"]

		avg_metrics_ft["precision"] += metrics_ft["precision"]
		avg_metrics_ft["recall"] += metrics_ft["recall"]
		avg_metrics_ft["f1"] += metrics_ft["f1"]

	avg_metrics_base = {k: v / total for k, v in avg_metrics_base.items()}
	avg_metrics_ft = {k: v / total for k, v in avg_metrics_ft.items()}

	return {
		"base_model": avg_metrics_base,
		"finetuned_model": avg_metrics_ft,
	}

def eval_oft_soft_matching(results: List[Dict]) -> Dict[str, Dict[str, float]]:
	total = len(results)

	avg_metrics_base = {"precision": 0, "recall": 0, "f1": 0}
	avg_metrics_ft = {"precision": 0, "recall": 0, "f1": 0}

	for res in results:
		metrics_base = calculate_re_metrics_soft_matching(
      		res["ground_truth"],
        	parse_json(res["base_pred"]).get("extracted_triplets", []) if parse_json(res["base_pred"]) is not None else [],
        	threshold=configs.GREEDY_MATCH_THRESHOLD,
		)
		metrics_ft = calculate_re_metrics_soft_matching(
      		res["ground_truth"],
        	parse_json(res["finetuned_pred"]).get("extracted_triplets", []) if parse_json(res["finetuned_pred"]) is not None else [],
         	threshold=configs.GREEDY_MATCH_THRESHOLD,
        )

		avg_metrics_base["precision"] += metrics_base["precision"]
		avg_metrics_base["recall"] += metrics_base["recall"]
		avg_metrics_base["f1"] += metrics_base["f1"]

		avg_metrics_ft["precision"] += metrics_ft["precision"]
		avg_metrics_ft["recall"] += metrics_ft["recall"]
		avg_metrics_ft["f1"] += metrics_ft["f1"]

	avg_metrics_base = {k: v / total for k, v in avg_metrics_base.items()}
	avg_metrics_ft = {k: v / total for k, v in avg_metrics_ft.items()}

	return {
		"base_model": avg_metrics_base,
		"finetuned_model": avg_metrics_ft,
	}

def eval_oft_semantic_matching(results: List[Dict]) -> Dict[str, Dict[str, float]]:
	# Load Sentence-BERT model for semantic evaluation
	print("Loading Sentence-BERT model...")
	sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
	print("Loading complete.")

	total = len(results)

	avg_metrics_base = {"precision": 0, "recall": 0, "f1": 0}
	avg_metrics_ft = {"precision": 0, "recall": 0, "f1": 0}

	for res in results:
		metrics_base = calculate_re_metrics_semantic(
      		res["ground_truth"],
        	parse_json(res["base_pred"]).get("extracted_triplets", []) if parse_json(res["base_pred"]) is not None else [],
			model=sbert_model,
         	threshold=configs.GREEDY_MATCH_THRESHOLD,
		)
		metrics_ft = calculate_re_metrics_semantic(
      		res["ground_truth"],
        	parse_json(res["finetuned_pred"]).get("extracted_triplets", []) if parse_json(res["finetuned_pred"]) is not None else [],
			model=sbert_model,
          	threshold=configs.GREEDY_MATCH_THRESHOLD,
        )

		avg_metrics_base["precision"] += metrics_base["precision"]
		avg_metrics_base["recall"] += metrics_base["recall"]
		avg_metrics_base["f1"] += metrics_base["f1"]

		avg_metrics_ft["precision"] += metrics_ft["precision"]
		avg_metrics_ft["recall"] += metrics_ft["recall"]
		avg_metrics_ft["f1"] += metrics_ft["f1"]

	avg_metrics_base = {k: v / total for k, v in avg_metrics_base.items()}
	avg_metrics_ft = {k: v / total for k, v in avg_metrics_ft.items()}

	return {
		"base_model": avg_metrics_base,
		"finetuned_model": avg_metrics_ft,
	}

def eval_oft():
	results = test_oft()

	em_metrics = eval_oft_exact_matching(results)

	sm_metrics = eval_oft_soft_matching(results)

	sem_metrics = eval_oft_semantic_matching(results)

	print("\nEvaluation Results:")
	for metrics, name in zip([em_metrics, sm_metrics, sem_metrics], ["Exact Matching", "Soft Matching", "Semantic Matching"]):
		print(f"Average Metrics ({name}) - Base Model: Precision={metrics['base_model']['precision']:.4f}, Recall={metrics['base_model']['recall']:.4f}, F1={metrics['base_model']['f1']:.4f}")
		print(f"Average Metrics ({name}) - Finetuned Model: Precision={metrics['finetuned_model']['precision']:.4f}, Recall={metrics['finetuned_model']['recall']:.4f}, F1={metrics['finetuned_model']['f1']:.4f}")

if __name__ == "__main__":
	eval_oft()
