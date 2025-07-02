import os
import math
import torch
import pandas as pd
import wandb
from modelscope import snapshot_download
from datasets import Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers import TextStreamer
from tqdm import tqdm
from datetime import datetime


# ✅ 数据加载
dataset_path = "/data/simple.csv"
df = pd.read_csv(dataset_path)

# 可选：填充 think 列
if 'think' not in df.columns:
    df['think'] = ''

# ✅ 构造 dataset
dataset = [
    {
        "instruction": row["prompt"],
        "input": "",
        "output": row["response"],
        "think": row["think"],
    }
    for _, row in df.iterrows()
]

# ✅ 划分训练/验证集
train_df, eval_df = train_test_split(
    pd.DataFrame(dataset), test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# ✅ 下载模型
model_name = os.getenv("DEFAULT_MODEL_NAME", "Qwen/Qwen3-1.7B")
model_dir = snapshot_download(model_name)
max_seq_length = 4096

# ✅ 自动生成文件名带时间戳
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# output_path = f"/data/test-results-{timestamp}.csv"

# 提取模型名称，替换掉可能存在的斜杠
# 例如，"Qwen/Qwen3-1.7B" 会变成 "Qwen3-1.7B"
model_info = model_name.split('/')[-1]

# ✅ 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# ✅ 格式化函数


def formatting_func_improved(examples):
    texts = []
    for i in range(len(examples["instruction"])):
        messages = [
            {"role": "system", "content": "你是一个编程助手，需要解决用户提出的技术问题。"},
            {"role": "user", "content": examples["instruction"][i]},
        ]
        think_step = examples["think"][i]
        output_step = examples["output"][i]
        assistant_content = f"<thinking>{think_step}</thinking>\n{output_step}" if think_step.strip(
        ) else output_step
        messages.append({"role": "assistant", "content": assistant_content})
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        texts.append(formatted)
    return {"text": texts}


train_dataset = train_dataset.map(formatting_func_improved, batched=True)
eval_dataset = eval_dataset.map(formatting_func_improved, batched=True)

# ✅ 应用 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.1,
    use_gradient_checkpointing=True,
    use_rslora=False,
    loftq_config=None,
)

# ✅ 网络代理（可选）
os.environ["HTTP_PROXY"] = "http://sing-box-clash:7890"
os.environ["HTTPS_PROXY"] = "http://sing-box-clash:7890"

# ✅ 登录 wandb
wandb.login()
run = wandb.init(project="lora_nocobase", name=f"test-run-{model_info}-{timestamp}")

# ✅ 自动计算 eval_steps


def compute_eval_steps(dataset_len, batch_size, grad_accum, epochs, evals_per_epoch=5):
    steps_per_epoch = math.ceil(dataset_len / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    return max(1, total_steps // (evals_per_epoch * epochs))


# ✅ 配置
batch_size = 1
grad_accum = 8
num_epochs = 15
eval_steps = compute_eval_steps(
    len(train_dataset), batch_size, grad_accum, num_epochs, evals_per_epoch=5)
print(f"[INFO] Auto eval_steps = {eval_steps}")

# ✅ 定义 Callback


class EvalCallback(TrainerCallback):
    def __init__(self, eval_steps):
        self.eval_steps = eval_steps

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            control.should_evaluate = True
        return control


# ✅ 启动训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_seq_length=max_seq_length,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=10,
        num_train_epochs=num_epochs,
        learning_rate=1e-5,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="wandb",
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
    ),
)

# ✅ 添加评估回调
trainer.add_callback(EvalCallback(eval_steps))

# ✅ 开始训练
trainer_stats = trainer.train()

# ✅ 保存结果
output_dir = "./output-simple"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)


# ✅ 读取测试集
test_df = pd.read_csv("/data/nocobase-qa-test.csv")
prompts = test_df["Prompt"].tolist()

# ✅ 设置生成参数
generation_kwargs = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# ✅ 推理生成回答
results = []
for prompt in tqdm(prompts, desc="Generating responses"):
    messages = [
        {"role": "system", "content": "你是一个编程助手，需要解决用户提出的技术问题。"},
        {"role": "user", "content": prompt},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **generation_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 精准保留 <think> 开始的回答正文部分
    thinking_index = response.find("<think>")
    if thinking_index != -1:
        response = response[thinking_index:].strip()
    else:
        system_intro = "你是一个编程助手，需要解决用户提出的技术问题。"
        if system_intro in response:
            response = response.split(system_intro)[-1].strip()
        if prompt in response:
            response = response.split(prompt)[-1].strip()

    results.append({
        "Prompt": prompt,
        "Response": response,
    })


output_path = f"/data/test-results-{model_info}-{timestamp}.csv"

# ✅ 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"[INFO] 推理结果已保存至 {output_path}")

results_artifact = wandb.Artifact(name=f"test_results_{model_info}", type="results",
                                  description=f"测试结果 for {model_info} 在 {timestamp}")
# 将 CSV 文件添加到工件
results_artifact.add_file(output_path)
run.log_artifact(results_artifact) # 这一步是请求上传工件

# 确保所有数据都被上传
wandb.finish()
print("CSV文件已作为工件上传到 Weights & Biases。")
