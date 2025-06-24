
from modelscope import snapshot_download

model_name = "Qwen/Qwen3-1.7B"
model_dir = snapshot_download(model_name)

model_name = "Qwen/Qwen3-8B"
model_dir = snapshot_download(model_name)

