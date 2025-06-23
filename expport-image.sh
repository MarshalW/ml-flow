#!/bin/bash

# 定义变量
IMAGE_NAME="ml-flow-ml-flow"  # 源镜像名称（根据docker image ls输出）
TAG="latest"                 # 镜像标签
OUTPUT_FILE="ml-flow.tar"    # 导出的tar文件名

# 检查Docker是否运行
if ! docker info >/dev/null 2>&1; then
    echo "错误：Docker服务未运行，请先启动Docker"
    exit 1
fi

# 检查镜像是否存在
if ! docker image inspect "${IMAGE_NAME}:${TAG}" >/dev/null 2>&1; then
    echo "错误：镜像 ${IMAGE_NAME}:${TAG} 不存在"
    echo "可用镜像列表："
    docker images
    exit 1
fi

# 导出镜像
echo "正在导出镜像 ${IMAGE_NAME}:${TAG} 到 ${OUTPUT_FILE}..."
if docker save -o "${OUTPUT_FILE}" "${IMAGE_NAME}:${TAG}"; then
    # 计算文件大小
    FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
    echo "导出成功！文件已保存为 ${OUTPUT_FILE} (大小: ${FILE_SIZE})"
    
    # 可选：验证文件完整性
    echo "正在验证文件完整性..."
    if tar tf "${OUTPUT_FILE}" >/dev/null; then
        echo "验证通过：tar文件结构完整"
    else
        echo "警告：tar文件可能损坏，建议重新导出"
        exit 1
    fi
else
    echo "错误：导出失败"
    exit 1
fi

# 可选：压缩文件（节省空间）
read -p "是否要压缩为gzip格式？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在压缩..."
    if gzip "${OUTPUT_FILE}"; then
        echo "压缩完成！文件已保存为 ${OUTPUT_FILE}.gz"
    else
        echo "压缩失败"
    fi
fi

exit 0