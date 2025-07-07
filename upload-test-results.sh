#!/bin/bash

for file in data/test-results-Qwen3-4B-*.csv; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        oss_path="oss://ml-lab-data/test-results/$filename"

        # 检查 OSS 是否已存在该文件
        if ossutil ls "$oss_path" > /dev/null 2>&1; then
            echo "文件已存在，跳过上传: $filename"
        else
            ossutil cp "$file" "$oss_path"
            echo "已上传: $filename"
        fi
    else
        echo "未找到匹配的文件。"
        exit 1
    fi
done