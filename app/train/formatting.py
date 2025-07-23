# formatting.py
def build_prompt(tokenizer, examples):
    system_content = """
你是专业的 NocoBase 开发助手，深度掌握 NocoBase 框架知识。请严格遵循以下准则：
1. 专注解答 NocoBase 插件开发、数据模型设计、API 扩展等问题
2. 对复杂问题提供分步解决方案和代码示例
3. 拒绝回答与 NocoBase 开发无关的请求
4. 所有涉及怎么做的响应必须包含可执行的代码片段或具体配置示例
"""
    texts = []
    for i in range(len(examples["instruction"])):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": examples["instruction"][i]},
        ]
        assistant_content = (
            f"<think>{examples['think'][i].strip()}</think>\n{examples['output'][i].strip()}"
            if examples["think"][i].strip()
            else examples["output"][i].strip()
        )
        messages.append({"role": "assistant", "content": assistant_content})
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    return {"text": texts}
