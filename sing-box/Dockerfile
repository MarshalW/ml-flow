FROM ghcr.io/sagernet/sing-box:latest

RUN apk add --no-cache curl yq jq \
    && mkdir -p /etc/sing-box

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
