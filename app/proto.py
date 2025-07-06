import os
import math
import torch
import pandas as pd
from modelscope import snapshot_download
from datasets import Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
from transformers import (
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer
from tqdm import tqdm
from datetime import datetime
import wandb


# ✅ 数据加载
dataset_path = "/data/datasets-20250703.csv"
df = pd.read_csv(dataset_path)
df["think"] = df.get("think", "").fillna("").astype(str)

dataset = [
    {
        "instruction": row["prompt"],
        "input": "",
        "output": row["response"],
        "think": row["think"],
    }
    for _, row in df.iterrows()
]

train_df, eval_df = train_test_split(pd.DataFrame(dataset), test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# ✅ 下载基础模型
model_name = os.getenv("DEFAULT_MODEL_NAME", "Qwen/Qwen3-1.7B")
model_dir = snapshot_download(model_name)
max_seq_length = 4096
model_info = model_name.split('/')[-1]

# ✅ 加载模型和分词器
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# ✅ 格式化数据
def formatting_func_clean(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        messages = [
            {"role": "system", "content": "你是一个编程助手，需要解决用户提出的技术问题。"},
            {"role": "user", "content": examples["instruction"][i]},
        ]
        think_step = examples["think"][i].strip()
        output_step = examples["output"][i].strip()
        assistant_content = f"<think>{think_step}</think>\n{output_step}" if think_step else output_step
        messages.append({"role": "assistant", "content": assistant_content})
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    return {"text": texts}

train_dataset = train_dataset.map(formatting_func_clean, batched=True)
eval_dataset = eval_dataset.map(formatting_func_clean, batched=True)

# ✅ 应用 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.1,
    use_gradient_checkpointing=True,
    use_rslora=False,
    loftq_config=None,
)

# ✅ 设置代理（保留）
os.environ["HTTP_PROXY"] = "http://sing-box-clash:7890"
os.environ["HTTPS_PROXY"] = "http://sing-box-clash:7890"

# ✅ 初始化 wandb
wandb.login()
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
run = wandb.init(project="lora_nocobase", name=f"test-run-{model_info}-{timestamp}")

# ✅ 超参配置
batch_size = 10
grad_accum = 1
num_epochs = 50
learning_rate = 5e-5
warmup_steps = 0

eval_steps = max(1, math.ceil(len(train_dataset) / (batch_size * grad_accum * 2)))
print(f"[INFO] Auto eval_steps = {eval_steps}")

# ✅ Trainer 参数
training_args = TrainingArguments(
    output_dir="./output-simple",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.001,
    lr_scheduler_type="cosine",
    seed=3407,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    bf16=True,
    report_to=["wandb"],
)

# ✅ 启动训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=max_seq_length,
    dataset_text_field="text",
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer_stats = trainer.train()

# ✅ 保存模型
output_dir = "./output-simple"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# ✅ 推理
test_df = pd.read_csv("/data/nocobase-qa-test.csv")
prompts = test_df["Prompt"].tolist()

generation_kwargs = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

results = []
for prompt in tqdm(prompts, desc="Generating responses"):
    messages = [
        {"role": "system", "content": "你是一个编程助手，需要解决用户提出的技术问题。"},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids_len = inputs.input_ids.shape[1]
    outputs = model.generate(**inputs, **generation_kwargs)
    generated_tokens = outputs[0][input_ids_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if "<think>" in response and "</think>" in response:
        content = response.split("<think>")[1].split("</think>")[0].strip()
        if not content:
            response = response.replace("<think></think>", "").strip()
    results.append({"Prompt": prompt, "Response": response})

# ✅ 保存推理结果
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
output_path = f"/data/test-results-{model_info}-{timestamp}.csv"
pd.DataFrame(results).to_csv(output_path, index=False)
print(f"[INFO] 推理结果已保存至 {output_path}")

artifact = wandb.Artifact(
    name=f"test_results_{model_info}",
    type="results",
    description=f"测试结果 for {model_info} at {timestamp}"
)
artifact.add_file(output_path)
run.log_artifact(artifact)