FROM ml-flow-base:latest

RUN /root/.venv/bin/pip install --no-cache-dir \
    wandb \
    scikit-learn
