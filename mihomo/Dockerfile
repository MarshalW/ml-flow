FROM alpine:3.20

# 安装必要工具
RUN apk add --no-cache curl bash tzdata && \
    mkdir -p /root/.config/clash

# 复制本地 Clash Meta 二进制文件
COPY mihomo-linux-amd64-v1.19.10.gz /usr/local/bin/clash-meta.gz
RUN gunzip /usr/local/bin/clash-meta.gz && \
    chmod +x /usr/local/bin/clash-meta

# 设置工作目录
WORKDIR /root/.config/clash

# 复制入口脚本和 GeoIP 数据库
COPY entrypoint.sh /entrypoint.sh
COPY config/Country.mmdb /root/.config/clash/Country.mmdb

# 设置环境变量
ENV SUBSCRIPTION_URL="https://connect.applecross.link/clash/xxx/xxxx" \
    CLASH_CONFIG_FILE="/root/.config/clash/config.yaml" \
    UPDATE_INTERVAL=86400

# 暴露代理端口
EXPOSE 7890 7891 9090

# 设置入口点
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]