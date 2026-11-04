#!/bin/bash
# ============================================================================
# RL training entrypoint — run one MARL trainer with Hydra overrides.
#
# Usage:
#   docker run --gpus all alem-rl <script> [hydra_overrides...]
#   docker run --gpus all -e RL_SCRIPT=mappo_rnn alem-rl [hydra_overrides...]
#
# Examples:
#   docker run --gpus all alem-rl ippo_rnn TOTAL_TIMESTEPS=1e4 NUM_ENVS=16 SEED=0
#   docker run --gpus all alem-rl mappo_rnn TRAINING_COORDINATION_DIFFICULTY=hard
# ============================================================================
set -euo pipefail

# Script name: from RL_SCRIPT env var, first positional arg, or show help.
SCRIPT="${RL_SCRIPT:-${1:-}}"

if [[ -z "$SCRIPT" || "$SCRIPT" == "--help" || "$SCRIPT" == "-h" ]]; then
    echo "Usage: docker run --gpus all alem-rl <script> [hydra_overrides...]"
    echo ""
    echo "Available training scripts:"
    for f in /app/baselines/*.py; do
        name="$(basename "$f" .py)"
        [[ "$name" == test* || "$name" == speed* || "$name" == utils ]] && continue
        echo "  $name"
    done
    echo ""
    echo "Common overrides:"
    echo "  TOTAL_TIMESTEPS=1e9  NUM_ENVS=1024  SEED=0"
    echo "  WANDB_MODE=online    ENTITY=<your-entity>  PROJECT=alem"
    echo "  TRAINING_COORDINATION_DIFFICULTY=easy|medium|hard"
    exit 0
fi

# If SCRIPT came in as a positional arg, drop it so the rest are Hydra overrides.
[[ "${1:-}" == "$SCRIPT" ]] && shift || true

echo "============================================"
echo " Script: baselines/${SCRIPT}.py"
echo " Args:   $*"
echo "============================================"

# Trainers find their Hydra config next to the script; run from /app so Hydra's
# outputs/ tree lands where Dockerfile.rl mounts it (-v ...:/app/outputs).
cd /app
exec python "baselines/${SCRIPT}.py" "$@"
