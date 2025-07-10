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
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "top_k": 40,
}

# ✅ 推理生成回答
results = []
system_content = """
你是专业的 NocoBase 开发助手，深度掌握 NocoBase 框架知识。请严格遵循以下准则：
1. 专注解答 NocoBase 插件开发、数据模型设计、API 扩展等问题
2. 对复杂问题提供分步解决方案和代码示例
3. 拒绝回答与 NocoBase 开发无关的请求
4. 所有涉及怎么做的响应必须包含可执行的代码片段或具体配置示例
"""
for prompt in tqdm(prompts, desc="Generating responses"):
    messages = [
        {"role": "system", "content": system_content},
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
    input_ids_len = inputs.input_ids.shape[1]

    outputs = model.generate(**inputs, **generation_kwargs)
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_tokens = outputs[0][input_ids_len:]  # 截掉 prompt 部分
    # 解码并清理输出
    response = tokenizer.decode(
        generated_tokens, skip_special_tokens=True).strip()
    
    # 去除多余 assistant
    if response.lower().startswith("assistant"):
        response = response[len("assistant"):].lstrip("：:").strip()


    # # 提取助手回复部分
    # assistant_start = response.find("<|assistant|>")
    # if assistant_start != -1:
    #     response = response[assistant_start + len("<|assistant|>"):].strip()

    # # 清理多余内容
    # if prompt in response:
    #     response = response.split(prompt)[-1].strip()

    results.append({
        "Prompt": prompt,
        "Response": response,
    })

# ✅ 保存结果
output_path = f"/data/test-results-{model_info}-{timestamp}.csv"
results_df = pd.DataFrame(results)
results_df.to_csv(output_path, index=False)
print(f"[INFO] 推理结果已保存至 {output_path}")
