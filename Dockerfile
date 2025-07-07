FROM ml-flow-base:latest

RUN /root/.venv/bin/pip install --no-cache-dir \
    wandb \
    scikit-learn

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    libssl-dev

RUN export http_proxy=http://myproxy:7890 && \
    export https_proxy=http://myproxy:7890 && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    unset http_proxy https_proxy