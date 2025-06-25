#!/bin/bash

# 检查订阅 URL 是否设置
if [ -z "$SUBSCRIPTION_URL" ]; then
  echo "Error: SUBSCRIPTION_URL is not set."
  exit 1
fi

# 下载配置文件函数
fetch_config() {
  echo "Fetching configuration from $SUBSCRIPTION_URL..."
  curl -L -o "$CLASH_CONFIG_FILE" "$SUBSCRIPTION_URL"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to fetch configuration."
    exit 1
  fi
  echo "Configuration fetched successfully."
}

# 初次下载配置文件
fetch_config

# 启动 Clash Meta
clash-meta -d /root/.config/clash &

# 定期更新配置文件
while true; do
  sleep "$UPDATE_INTERVAL"
  fetch_config
  # 重启 Clash Meta 以应用新配置
  pkill clash-meta
  clash-meta -d /root/.config/clash &
done