#!/bin/bash
# ============================================================================
# LLM eval entrypoint: start a vLLM server, wait until it's ready, then run a
# 3-agent Alem evaluation against it. One container does everything.
#
# Usage:
#   docker run --gpus all --shm-size=16g -e MODEL_ID=<hf-model> alem-llm [eval_overrides...]
#
# Example:
#   docker run --gpus all --shm-size=16g \
#     -e MODEL_ID=meta-llama/Llama-3.2-1B-Instruct \
#     -e HF_TOKEN=<token> \
#     -v ~/.cache/huggingface:/app/.cache/huggingface \
#     alem-llm eval.num_episodes.alem=5
#
#   # Connect to an already-running vLLM server instead of starting one:
#   docker run -e MODEL_ID=<model> -e SKIP_VLLM=1 -e VLLM_PORT=8000 alem-llm
# ============================================================================
set -euo pipefail

MODEL_ID="${MODEL_ID:?ERROR: MODEL_ID is required (e.g. meta-llama/Llama-3.2-1B-Instruct)}"

PORT="${VLLM_PORT:-8080}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
SKIP_VLLM="${SKIP_VLLM:-0}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
BASE_URL="http://localhost:${PORT}/v1"

# Default to single-GPU; callers override sharding via VLLM_EXTRA_ARGS.
if [[ "${VLLM_EXTRA_ARGS:-}" != *"--tensor-parallel-size"* ]]; then
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-} --tensor-parallel-size 1"
fi

echo "============================================"
echo " Model:    ${MODEL_ID}"
echo " Base URL: ${BASE_URL}"
echo " GPU util: ${GPU_UTIL}   Max len: ${MAX_MODEL_LEN}"
echo " vLLM args:${VLLM_EXTRA_ARGS}"
echo "============================================"

# ── HuggingFace cache + auth ──────────────────────────────────────────────────
export HF_HOME="${HF_HOME:-/app/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
if [[ -n "${HF_TOKEN:-}" ]]; then
    mkdir -p "${HF_HOME}"
    echo "${HF_TOKEN}" > "${HF_HOME}/token"
fi

# ── Process management: stop vLLM when the eval finishes or the container dies ─
VLLM_PID=""
TAIL_PID=""
cleanup() {
    [[ -n "$TAIL_PID" ]] && kill "$TAIL_PID" 2>/dev/null || true
    if [[ -n "$VLLM_PID" ]]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        kill -- -"$VLLM_PID" 2>/dev/null || kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── Start vLLM server ─────────────────────────────────────────────────────────
if [[ "$SKIP_VLLM" != "1" ]]; then
    mkdir -p /app/logs/vllm
    VLLM_LOG="/app/logs/vllm/vllm_${PORT}.log"

    echo "Starting vLLM server..."
    # setsid: new process group so `kill -- -PID` reaches vLLM's worker children.
    # shellcheck disable=SC2086  # VLLM_EXTRA_ARGS intentionally word-splits
    setsid vllm serve "$MODEL_ID" \
        --port "$PORT" \
        --gpu-memory-utilization "$GPU_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        ${VLLM_EXTRA_ARGS} \
        >> "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!

    tail -f "$VLLM_LOG" &   # mirror server log to `docker logs`
    TAIL_PID=$!

    echo "Waiting for vLLM to be ready (timeout: ${STARTUP_TIMEOUT}s)..."
    for i in $(seq 1 "$STARTUP_TIMEOUT"); do
        if curl -sf "${BASE_URL}/models" > /dev/null 2>&1; then
            echo "vLLM server ready after ${i}s."
            break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM server exited unexpectedly. Last 60 log lines:"
            tail -60 "$VLLM_LOG" 2>/dev/null || true
            exit 1
        fi
        (( i % 30 == 0 )) && echo "  ... still waiting (${i}s / ${STARTUP_TIMEOUT}s)"
        sleep 1
    done

    if ! curl -sf "${BASE_URL}/models" > /dev/null 2>&1; then
        echo "ERROR: vLLM did not become ready within ${STARTUP_TIMEOUT}s."
        exit 1
    fi
else
    echo "SKIP_VLLM=1 — connecting to existing server at ${BASE_URL}"
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
# All 3 agents share the one vLLM server / model. Extra Hydra overrides can be
# passed via EVAL_EXTRA_ARGS or as trailing container args.
echo ""
echo "Running LLM evaluation..."
echo "============================================"
cd /app
# shellcheck disable=SC2086  # EVAL_EXTRA_ARGS intentionally word-splits
exec python baselines/llm/eval_alem.py \
    "clients.0.client_name=vllm" "clients.0.model_id=${MODEL_ID}" "clients.0.base_url=${BASE_URL}" \
    "clients.1.client_name=vllm" "clients.1.model_id=${MODEL_ID}" "clients.1.base_url=${BASE_URL}" \
    "clients.2.client_name=vllm" "clients.2.model_id=${MODEL_ID}" "clients.2.base_url=${BASE_URL}" \
    "WANDB_MODE=${WANDB_MODE:-disabled}" \
    ${EVAL_EXTRA_ARGS:-} \
    "$@"
