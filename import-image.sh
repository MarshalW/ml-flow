#!/bin/bash

TAR_FILE="ml-flow.tar.gz"
IMAGE_NAME="ml-flow-ml-flow"
TAG="latest"

# 检查文件是否存在
if [ ! -f "$TAR_FILE" ]; then
    echo "错误：文件 $TAR_FILE 不存在" >&2
    exit 1
fi

# 导入镜像
echo "正在导入镜像..."
if [[ "$TAR_FILE" == *.gz ]]; then
    gunzip -c "$TAR_FILE" | docker load
else
    docker load -i "$TAR_FILE"
fi

# 获取最新导入的镜像ID
NEW_IMAGE_ID=$(docker images -q | head -n 1)

if [ -z "$NEW_IMAGE_ID" ]; then
    echo "错误：镜像导入失败" >&2
    exit 1
fi

# 重命名镜像
echo "重命名镜像为 $IMAGE_NAME:$TAG..."
docker tag "$NEW_IMAGE_ID" "$IMAGE_NAME:$TAG"

# 验证
if docker inspect "$IMAGE_NAME:$TAG" >/dev/null 2>&1; then
    echo "成功导入镜像:"
    docker images | grep "$IMAGE_NAME"
else
    echo "镜像重命名失败" >&2
    exit 1
fi