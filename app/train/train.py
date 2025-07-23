# train.py
from config import *
from data import load_dataset
from formatting import build_prompt
from unsloth import FastLanguageModel
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
import wandb
import os, shutil
import math
from modelscope import snapshot_download
import torch


def main():
    # 下载基础模型
    model_dir = snapshot_download(MODEL_NAME)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # 数据加载
    train_dataset, eval_dataset = load_dataset(DATASET_PATH)
    train_dataset = train_dataset.map(
        lambda x: build_prompt(tokenizer, x), batched=True
    )
    eval_dataset = eval_dataset.map(lambda x: build_prompt(tokenizer, x), batched=True)

    # LoRA 注入
    model = FastLanguageModel.get_peft_model(
        model,
        r=4,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=8,
        lora_dropout=0.05,
        use_gradient_checkpointing=True,
    )

    # 环境配置
    os.environ["HTTP_PROXY"] = "http://sing-box-clash:7890"
    os.environ["HTTPS_PROXY"] = "http://sing-box-clash:7890"

    # wandb 初始化
    wandb.login()
    wandb.init(project=WANDB_PROJECT, name=WANDB_NAME)

    # Trainer 配置
    eval_steps = max(1, math.ceil(len(train_dataset) / (BATCH_SIZE * GRAD_ACCUM * 2)))
    training_args = TrainingArguments(
        output_dir=CHECKPOINTS_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=SEED,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        report_to=["wandb"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # 模型保存
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if os.path.exists(CHECKPOINTS_DIR):
        shutil.rmtree(CHECKPOINTS_DIR)


if __name__ == "__main__":
    main()
