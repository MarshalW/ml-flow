import os
import torch
import pandas as pd
from modelscope import snapshot_download
from unsloth import FastLanguageModel
from tqdm import tqdm
from datetime import datetime
from peft import PeftModel


# ✅ 模型和适配器配置
base_model_name = os.getenv("DEFAULT_MODEL_NAME", "Qwen/Qwen3-1.7B")  # 基座模型名称

model_info = base_model_name.split('/')[-1]
lora_adapter_path = f"./output-adapter-{model_info}"

# lora_adapter_path = "./output-simple"  # LoRA适配器保存路径
max_seq_length = 4096  # 与训练时保持一致

# ✅ 下载基座模型
base_model_dir = snapshot_download(base_model_name)

# ✅ 加载基座模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_dir,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

# ✅ 加载LoRA适配器
model = PeftModel.from_pretrained(model, lora_adapter_path)
model = FastLanguageModel.for_inference(model)  # 优化推理速度

# ✅ 创建时间戳和模型信息
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
# model_info = base_model_name.split('/')[-1]

# ✅ 读取测试集
test_df = pd.read_csv("/data/nocobase-qa-test.csv")
prompts = test_df["Prompt"].tolist()

# ✅ 设置生成参数（与训练时一致）
generation_kwargs = {
    "max_new_tokens": 1024,
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
    
    # 格式化输入
    formatted = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 生成响应
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **generation_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取助手回复部分
    assistant_start = response.find("<|assistant|>")
    if assistant_start != -1:
        response = response[assistant_start + len("<|assistant|>"):].strip()
    
    # 清理多余内容
    if prompt in response:
        response = response.split(prompt)[-1].strip()
    
    results.append({
        "Prompt": prompt,
        "Response": response,
    })

# ✅ 保存结果
output_path = f"/data/test-results-{model_info}-{timestamp}.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"[INFO] 推理结果已保存至 {output_path}")