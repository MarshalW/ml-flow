# config.py
import os
from datetime import datetime

MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "Qwen/Qwen3-1.7B")
CHECKPOINTS_DIR = "./output-checkpoints"
OUTPUT_DIR = f"./output-adapter-{MODEL_NAME.split('/')[-1]}"
# DATASET_PATH = "/data/datasets-20250710.csv"
DATASET_PATH = "/data/datasets-20250722.csv"

MAX_SEQ_LENGTH = 4096
BATCH_SIZE = 5
GRAD_ACCUM = 3
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5 # 5e-5
WARMUP_STEPS = 0
SEED = 3407

WANDB_PROJECT = "lora_nocobase"
WANDB_NAME = (
    f"test-run-{MODEL_NAME.split('/')[-1]}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
)
