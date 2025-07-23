# eval.py
import os
import torch
import pandas as pd
from modelscope import snapshot_download
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm import tqdm
from datetime import datetime
from config import MODEL_NAME, MAX_SEQ_LENGTH
from transformers import set_seed


def generate_responses(test_csv_path, output_dir="/data"):
    # ✅ 代理设置（可选）
    os.environ["HTTP_PROXY"] = "http://sing-box-clash:7890"
    os.environ["HTTPS_PROXY"] = "http://sing-box-clash:7890"

    # ✅ 加载模型和 tokenizer
    base_model_info = MODEL_NAME.split("/")[-1]
    base_model_dir = snapshot_download(MODEL_NAME)
    lora_adapter_path = f"./output-adapter-{base_model_info}"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(model, lora_adapter_path)
    model = FastLanguageModel.for_inference(model)

    # ✅ system prompt（与训练保持一致）
    system_content = """
你是专业的 NocoBase 开发助手，深度掌握 NocoBase 框架知识。请严格遵循以下准则：
1. 专注解答 NocoBase 插件开发、数据模型设计、API 扩展等问题
2. 对复杂问题提供分步解决方案和代码示例
3. 拒绝回答与 NocoBase 开发无关的请求
4. 所有涉及怎么做的响应必须包含可执行的代码片段或具体配置示例
"""

    # ✅ 推理参数
    generation_kwargs = {
        "max_new_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "top_k": 40,
    }

    # ✅ 加载测试集
    df = pd.read_csv(test_csv_path)
    prompts = df["Prompt"].tolist()

    # ✅ 推理生成
    results = []
    for prompt in tqdm(prompts, desc="Generating responses"):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_ids_len = inputs.input_ids.shape[1]

        outputs = model.generate(**inputs, **generation_kwargs)
        generated_tokens = outputs[0][input_ids_len:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if response.lower().startswith("assistant"):
            response = response[len("assistant") :].lstrip("：:").strip()

        results.append(
            {
                "Prompt": prompt,
                "Response": response,
            }
        )

    # ✅ 保存结果
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_path = os.path.join(
        output_dir, f"test-results-{base_model_info}-{timestamp}.csv"
    )
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"[INFO] 推理结果已保存至 {output_path}")


def main():
    set_seed(42)
    test_csv_path = "/data/nocobase-qa-test.csv"
    output_dir = "/data"
    generate_responses(test_csv_path, output_dir)


if __name__ == "__main__":
    main()
