import os

# ===================================== #
# Basic Configurations                  #
# ===================================== #
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_DIR = os.path.join("models", os.path.basename(MODEL_ID).lower())

TRAIN_PATH = "data/train.json"
VALID_PATH = "data/valid.json"

LOG_CHECKPOINTS_DIR = "./qwen-oft-relation-extraction"
FINETUNED_MODEL_DIR = "./qwen-oft-final"

EVALUATION_RESULTS_PATH = "eval_oft_results.jsonl"

# ===================================== #
# Configuration Settings for Training   #
# ===================================== #

# BOFT Configuration
BOFT_BLOCK_SIZE = 4
BOFT_N_BUTTERFLY_FACTOR = 2
BOFT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
BOFT_DROPOUT = 0.1

# Training Hyperparameters
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1

# ===================================== #
# Configuration Settings for Evaluation #
# ===================================== #
EXACT_MATCH_TYPES = {"subject", "predicate", "object"}
GREEDY_MATCH_THRESHOLD = 0.66
