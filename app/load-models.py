
import os
from modelscope import snapshot_download

snapshot_download("Qwen/Qwen3-1.7B")
snapshot_download(os.getenv("DEFAULT_MODEL_NAME", "Qwen/Qwen3-1.7B"))

