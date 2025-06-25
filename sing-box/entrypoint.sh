#!/bin/sh

CLASH_SUB_URL="${CLASH_SUB_URL:-}"
PROXY_NAME="${PROXY_NAME:-🇯🇵 Relay-JP2}"
UA="Clash/1.8.0"

if [ -z "$CLASH_SUB_URL" ]; then
  echo "Error: CLASH_SUB_URL is not set."
  exit 1
fi

echo "Downloading Clash config from: $CLASH_SUB_URL"
curl -sSL -H "User-Agent: ${UA}" "$CLASH_SUB_URL" -o /root/clash_config.yaml

echo "Generating sing-box config..."
mkdir -p /etc/sing-box

# 🔥 将 yq 表达式用双引号包裹，$PROXY_NAME 变量会被展开
yq eval -o=json "
  .proxies[] | select(.name == \"$PROXY_NAME\") |
  {
    \"outbounds\": [
      {
        \"type\": \"shadowsocks\",
        \"tag\": \"ss-out\",
        \"method\": .cipher,
        \"password\": .password,
        \"server\": .server,
        \"server_port\": .port,
        \"udp_over_tcp\": true
      },
      {
        \"type\": \"direct\",
        \"tag\": \"direct\"
      },
      {
        \"type\": \"block\",
        \"tag\": \"block\"
      }
    ],
    \"inbounds\": [
      {
        \"type\": \"http\",
        \"listen\": \"0.0.0.0\",
        \"listen_port\": 7890,
        \"tag\": \"http-in\"
      }
    ],
    \"route\": {
      \"rules\": [
        {
          \"ip_cidr\": [\"0.0.0.0/0\"],
          \"outbound\": \"ss-out\"
        }
      ]
    }
  }
" /root/clash_config.yaml > /etc/sing-box/config.json

echo "[DEBUG] Generated config.json content:"
cat /etc/sing-box/config.json

# 验证 JSON
if ! jq empty /etc/sing-box/config.json > /dev/null 2>&1; then
  echo "[ERROR] config.json is not valid JSON"
  exit 1
fi

echo "Starting sing-box..."
exec sing-box run -c /etc/sing-box/config.json
