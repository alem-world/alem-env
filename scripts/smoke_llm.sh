#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/smoke_llm.sh MODEL_ID [options]

Runs a tiny 3-agent Alem smoke test against an OpenAI-compatible API.
Use this before a full leaderboard run.

Example:
  scripts/smoke_llm.sh TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --base-url http://localhost:8000/v1 --steps 5 --coord easy

Options:
  --base-url URL       OpenAI-compatible /v1 endpoint. Default: http://localhost:8000/v1
  --api-key KEY        API key. Default: EMPTY for local endpoints, or OPENAI_API_KEY.
  --steps N           Number of environment steps. Default: 5
  --coord LEVEL        none, easy, medium, or hard. Default: easy
  --temperature X     Sampling temperature. Default: 0.2
  --max-tokens N      Max tokens per agent call. Default: 128
  --help              Show this message.
USAGE
}

if [[ $# -lt 1 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

MODEL_ID="$1"
shift

BASE_URL="http://localhost:8000/v1"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
STEPS="5"
COORD="easy"
TEMPERATURE="0.2"
MAX_TOKENS="128"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --coord) COORD="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

exec uv run --extra llm --python 3.12 python examples/llm_openai_smoke.py \
  --model "${MODEL_ID}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}" \
  --steps "${STEPS}" \
  --coord "${COORD}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}"
