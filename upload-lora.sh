#!/bin/bash

# 动态查找符合模式的目录（以 output-adapter-Qwen3- 开头）
CANDIDATES=(app/output-adapter-Qwen3-*/)

# 检查是否只有一个匹配项
if [ ${#CANDIDATES[@]} -ne 1 ]; then
    echo "错误：找到多个或没有匹配的目录。请确保只有一个 output-adapter-Qwen3-* 目录。"
    exit 1
fi

# 设置源目录路径
SOURCE_DIR="${CANDIDATES[0]}"

# 提取模型标识符（如 4b, 8b），并转为小写
MODEL_SUFFIX=$(basename "$SOURCE_DIR" | awk -F '-' '{print tolower($NF)}')  # 参考了 awk 的使用 [[1]]

# 获取当前时间戳，格式为 YYYYMMDDHHMM
TIMESTAMP=$(date +"%Y%m%d%H%M")  # 使用 date 命令获取时间戳 [[9]]

# 目标压缩包名称
OUTPUT_NAME="lora-adapter-qwen3-${MODEL_SUFFIX}-${TIMESTAMP}.tar.gz"

# 打包命令
tar -czf "$OUTPUT_NAME" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"  # tar 打包方式参考 [[8]]

echo "打包完成: $OUTPUT_NAME"

# 上传到 OSS
ossutil cp "$OUTPUT_NAME" "oss://ml-lab-data/lora-adapters/$OUTPUT_NAME"

echo "上传完成: $OUTPUT_NAME"

# 删除本地文件
rm "$OUTPUT_NAME"