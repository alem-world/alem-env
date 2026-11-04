#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_llm_eval.sh MODEL_ID [options]

Runs Alem LLM evaluation against an OpenAI-compatible API such as vLLM.
The same MODEL_ID is used for all three agents.

Common examples:
  # 5-step local vLLM smoke test
  scripts/run_llm_eval.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --base-url http://localhost:8000/v1 --episodes 1 --steps 5 --difficulty easy --smoke

  # Submission run on all leaderboard difficulties
  scripts/run_llm_eval.sh meta-llama/Llama-3.2-1B-Instruct \
    --base-url http://localhost:8000/v1 --episodes 20 --difficulty easy,medium,hard

Options:
  --base-url URL       OpenAI-compatible /v1 endpoint. Default: http://localhost:8000/v1
  --client NAME       Client backend: vllm, openai, nvidia, xai. Default: vllm
  --episodes N        Episodes per difficulty. Default: 20
  --steps N           Max steps per episode. Default: 10000
  --difficulty LIST   easy, medium, hard, or comma list. Default: easy,medium,hard
  --workers N         Eval workers. Default: 1
  --agent TYPE        Agent harness. Default: robust_all
  --smoke             Disable expensive artifacts/debriefs and W&B; intended for quick checks.
  --help              Show this message.

Environment:
  WANDB_MODE=disabled is recommended unless you explicitly want W&B logging.
  For OpenAI, set OPENAI_API_KEY and use --client openai.
  For vLLM, no API key is required.
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

MODEL_ID="$1"
shift

BASE_URL="http://localhost:8000/v1"
CLIENT="vllm"
EPISODES="20"
STEPS="10000"
DIFFICULTY="easy,medium,hard"
WORKERS="1"
AGENT="robust_all"
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --client) CLIENT="$2"; shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --difficulty) DIFFICULTY="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --agent) AGENT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

EXTRA=(
  "agent.type=${AGENT}"
  "alem.coordination_difficulty=${DIFFICULTY}"
  "eval.num_episodes.alem=${EPISODES}"
  "eval.max_steps_per_episode=${STEPS}"
  "eval.num_workers=${WORKERS}"
  "clients.0.client_name=${CLIENT}"
  "clients.1.client_name=${CLIENT}"
  "clients.2.client_name=${CLIENT}"
  "clients.0.model_id=${MODEL_ID}"
  "clients.1.model_id=${MODEL_ID}"
  "clients.2.model_id=${MODEL_ID}"
)

if [[ "${CLIENT}" != "openai" ]]; then
  EXTRA+=(
    "clients.0.base_url=${BASE_URL}"
    "clients.1.base_url=${BASE_URL}"
    "clients.2.base_url=${BASE_URL}"
  )
fi

if [[ "${SMOKE}" == "1" ]]; then
  export WANDB_MODE="${WANDB_MODE:-disabled}"
  EXTRA+=(
    "WANDB_MODE=disabled"
    "eval.save_images=false"
    "eval.debug=false"
    "eval.generate_debriefs=false"
    "agent.max_image_history=0"
    "agent.max_text_history=4"
    "clients.0.generate_kwargs.max_tokens=128"
    "clients.1.generate_kwargs.max_tokens=128"
    "clients.2.generate_kwargs.max_tokens=128"
  )
fi

if [[ "${DIFFICULTY}" == *","* ]]; then
  exec uv run --extra baselines-llm --python 3.12 python baselines/llm/eval_alem.py -m "${EXTRA[@]}"
else
  exec uv run --extra baselines-llm --python 3.12 python baselines/llm/eval_alem.py "${EXTRA[@]}"
fi
