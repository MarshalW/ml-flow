# ML Flow

Machine Learning Flow, 用于 LLM 迭代微调.

## 本地迭代过程

```bash

docker exec -it ml-flow bash

# 训练
time python proto-train.py

# 测试
time python proto-test.py

# 保存 gguf 模型
time python ./proto-save-gguf.py

# 上传模型到 Ollama
cd model
time ./push.sh


# 退出容器

# 上传 lora adapter
time ./upload-lora.sh

# 上传测试结果
time ./upload-test-results.sh
```