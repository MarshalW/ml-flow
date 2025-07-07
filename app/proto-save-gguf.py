import os
import torch
from modelscope import snapshot_download
from unsloth import FastLanguageModel
from datetime import datetime
from peft import PeftModel
import shutil

# ✅ 模型和适配器配置
base_model_name = os.getenv("DEFAULT_MODEL_NAME", "Qwen/Qwen3-1.7B")  # 基座模型名称
model_info = base_model_name.split('/')[-1]
lora_adapter_path = f"./output-adapter-{model_info}"
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

# 第一次使用时自动 git clone llama.cpp@github
os.environ['http_proxy'] = 'http://sing-box-clash:7890'
os.environ['https_proxy'] = 'http://sing-box-clash:7890'

destination_dir = "./model"
temp_dir = "/tmp/gguf"
source_file=f"{temp_dir}/unsloth.Q4_K_M.gguf"

model.save_pretrained_gguf(temp_dir, tokenizer, quantization_method="q4_k_m")

try:
    shutil.copy2(source_file, destination_dir)
    print("文件复制成功。")

finally:
    shutil.rmtree(temp_dir)
