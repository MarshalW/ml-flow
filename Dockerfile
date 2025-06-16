# 使用 Ubuntu 24.04 官方镜像作为基础
FROM ubuntu:24.04

# 设置非交互式环境变量（避免apt提示中断构建）
ENV DEBIAN_FRONTEND=noninteractive