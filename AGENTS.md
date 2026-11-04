# AGENTS.md

Instructions for coding agents working in this repo. Humans: start with [`README.md`](README.md).

Alem is a JAX benchmark for **open-ended multi-agent coordination**. The common task here is:
serve an LLM, run a 3-agent team through the text interface, and get a comparable score.

## The protocol is fixed, so don't improvise it

A score is only comparable if it was produced under the standard settings. These are the
defaults; change one and you no longer have a leaderboard number, you have an ablation.

| | Value |
| --- | --- |
| Harness | `robust_all` + `specific_collaborative` prompt (CoT, communication, scratchpad) |
| Team | 3 zero-shot homogeneous agents, one model driving all three |
| Difficulties | `easy`, `medium`, `hard`. **Reported separately, never averaged** |
| Episodes | 20 per difficulty (10 minimum if cost-bound), shared seeds (`EVAL_SEED=9999`) |
| Native reasoning | `agent.reasoning=true`. **Not the default**, see Traps |

Full protocol: [`EVALUATION.md`](EVALUATION.md). Submission rules: [`SUBMISSION.md`](SUBMISSION.md).

## Setup: two environments, not one

vLLM pins its own torch/CUDA build, which clashes with this repo's jax. Installing both into
one environment is the single most common way to lose an afternoon.

```bash
uv pip install -e ".[baselines-llm]"        # this repo: eval harness (jax, CPU is fine)

uv venv --python 3.12 .venv-vllm            # a SEPARATE env, only for serving
uv pip install --python .venv-vllm vllm --torch-backend=auto
```

## Run it

```bash
# 1. serve (separate env, own terminal)
.venv-vllm/bin/vllm serve MODEL_ID --port 8000 --max-model-len 32768

# 2. smoke-test the wiring before spending tokens: 5 steps, one episode
scripts/smoke_llm.sh MODEL_ID --base-url http://localhost:8000/v1 --steps 5 --coord easy

# 3. the real thing (defaults: 20 episodes, all three difficulties)
scripts/run_llm_eval.sh MODEL_ID --base-url http://localhost:8000/v1 \
    --episodes 20 --difficulty easy,medium,hard
```

Hosted APIs need no vLLM: pass `--client openai|anthropic|gemini|nvidia|xai` and the matching
key. Results land in `outputs/alem_eval/<timestamp>_<model>_<difficulty>/`, one
`alem/default/default_run_NN.json` per episode plus `summary_stats.json`.

## Turn the run into a submission

```bash
python scripts/make_submission.py --model-id MODEL_ID --type open-weight \
    --family FAMILY --params "31B dense"
```

This prints the leaderboard entry (Base% / Coord.% / Total%, each with an rliable bootstrap
95% CI) and writes a verification bundle. **Use it. Do not compute the three numbers by
hand.** Each episode json carries several similarly-named metric families, and the leaderboard
triple comes from exactly one of them; picking the wrong one silently halves every score.

## Traps that have cost real time

- **Native reasoning is off unless you ask for it.** `agent.reasoning` defaults to null, which
  resolves to `False` (`baselines/llm/eval_utils/agents/__init__.py:113`). Every leaderboard
  entry was run with `agent.reasoning=true`. On vLLM it also needs a matching
  `--reasoning-parser` on the server, or the reasoning arrives inline and the action parser
  trips over it.
- **Check `action_parse_rate` before believing a low score.** It is in every episode json. A
  rate well under ~0.95 means the score is throttled by output formatting, not by the model's
  ability. Fix the parsing before reporting the number.
- **`rliable` needs `arch < 8` and `pandas < 3`.** Newer `arch` dropped the `random_state`
  argument `rliable` still passes, and `make_submission.py` dies in the bootstrap. Pin them in
  whatever env you run the submission from.
- **Never average difficulties.** `easy`/`medium`/`hard` differ in the coordination weight
  (alpha 0.30 / 0.60 / 0.90) and measure different things.
- **Episodes are long.** `max_steps_per_episode` is 10000, but episodes end when all agents
  die, typically after a few hundred steps. Budget wall-clock accordingly; a 20-episode difficulty
  against a large local model is hours, not minutes.

## Conventions

- Package manager is `uv`. Python 3.12.
- Lint/format with `ruff` (config in `pyproject.toml`); tests with `pytest`.
- Don't commit anything under `outputs/`.
