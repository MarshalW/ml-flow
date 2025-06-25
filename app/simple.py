import pandas as pd
from modelscope import snapshot_download
from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

dataset_path = "/data/simple.csv"
model_name = "Qwen/Qwen3-1.7B"
max_seq_length = 4096
output_dir = "./output-simple"

df = pd.read_csv(dataset_path)
df.insert(len(df.columns), 'think', '')  # 临时用空的列

dataset = []

for _, row in df.iterrows():
    dataset.append({
        "instruction": row["prompt"],  # 注意列名拼写检查！
        "input": "",
        "output": row["response"],
    })


model_dir = snapshot_download(model_name)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=max_seq_length,  # 根据硬件调整
    load_in_4bit=False,  # 禁用4位量化
    dtype=torch.bfloat16,
    device_map="auto",    # 多GPU自动分配
    attn_implementation="flash_attention_2",  # 启用 FlashAttention-2
)


def formatting_func_improved(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        messages = [
            {"role": "system", "content": "你是一个编程助手，需要解决用户提出的技术问题。"},
            {"role": "user", "content": examples["instruction"][i]}
        ]
        assistant_content = ""
        think_step = examples["think"][i] if "think" in examples and i < len(
            examples["think"]) else ""
        output_step = examples["output"][i]
        if think_step and think_step.strip():
            assistant_content = f"<thinking>{think_step}</thinking>\n{output_step}"
        else:
            assistant_content = output_step
        messages.append({"role": "assistant", "content": assistant_content})
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(formatted_text)
    return {"text": texts}


dataset = Dataset.from_pandas(pd.DataFrame(dataset))
dataset = dataset.map(formatting_func_improved, batched=True)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # 对于8B模型可以尝试16或32
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.1,
    use_gradient_checkpointing=True,
    # 添加QLoRA特定参数
    use_rslora=False,  # 是否使用Rank-Stabilized LoRA
    loftq_config=None,  # 可以配置LoftQ初始化
)

import os
os.environ["HTTP_PROXY"] = "http://sing-box-clash:7890"
os.environ["HTTPS_PROXY"] = "http://sing-box-clash:7890"

import wandb
wandb.login()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,  # Can set up evaluation!
    max_seq_length=max_seq_length,  # 自己加的
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        # num_train_epochs = 1, # Set this for 1 full training run.
        num_train_epochs=15,
        # max_steps = 30,
        learning_rate=1e-5,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        # report_to="none",  # Use this for WandB etc
        report_to="wandb",
        # 新增对齐旧的调用参数
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
    ),
)

trainer_stats = trainer.train()
