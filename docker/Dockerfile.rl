# ============================================================================
# alem — RL training image (JAX / CUDA 12)
#
# Runs the JAX MARL trainers (IPPO / MAPPO / PQN-VDN / HyperMARL) on GPU.
# Requires nvidia-container-toolkit on the host and --gpus at runtime.
#
# Build (from repo root):
#   docker build -f docker/Dockerfile.rl -t alem-rl .
#
# Run:
#   docker run --rm --gpus all alem-rl                       # prints usage
#   docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" \
#       alem-rl ippo_rnn TOTAL_TIMESTEPS=1e4 NUM_ENVS=16 SEED=0
#
# Env vars: RL_SCRIPT (script name, no .py) · WANDB_API_KEY · WANDB_MODE
#           (default offline) · XLA_PYTHON_CLIENT_MEM_FRACTION (default 0.95)
# ============================================================================
ARG CUDA_VERSION=12.9.0
ARG PYTHON_VERSION=3.12

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04
ARG PYTHON_VERSION

LABEL org.opencontainers.image.title="alem-rl" \
      org.opencontainers.image.description="JAX MARL training for Alem (IPPO/MAPPO/PQN-VDN/HyperMARL)"

ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    WANDB_MODE=offline \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Ubuntu 24.04 ships Python 3.12 natively.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
        build-essential curl git libgl1 libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
# [baselines-rl] = MARL trainer deps; [gpu] = jax[cuda12] wheels for GPU kernels.
RUN pip install --no-cache-dir -e ".[baselines-rl,gpu]"

COPY docker/entrypoint.rl.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--help"]
