# Submit to the Alem leaderboard

*Alem* is an open benchmark: take any model, get a comparable number, and add it to the
[leaderboard](https://alem-world.github.io/leaderboard.html) in three commands. The
canonical protocol (seeds, episodes, metrics) is in [`EVALUATION.md`](EVALUATION.md); the
default LLM harness is `robust_all` — three zero-shot agents with communication, scratchpad
memory, and reasoning.

## LLM track

Works with any OpenAI-compatible endpoint plus Anthropic and Gemini. Pick your provider:

| Provider | `--client` | Example `MODEL_ID` | API key env var |
| --- | --- | --- | --- |
| Local open weights (vLLM) | `vllm` (default) | `Qwen/Qwen3.5-9B`, `meta-llama/Llama-3.3-70B-Instruct` | — (local server) |
| OpenAI | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini` | `gemini-3.1-pro-preview` | `GEMINI_API_KEY` |
| NVIDIA NIM / xAI | `nvidia` / `xai` | `meta/llama-3.3-70b-instruct` | `NVIDIA_API_KEY` (+ `--base-url`) |

**1. Local models only — serve with vLLM** (needs its own env; see
[README → Evaluate an LLM](README.md#evaluate-an-llm), or use the
[`alem-llm` Docker image](README.md#docker) which does serve + eval in one `docker run`):

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000 --max-model-len 32768
```

**2. Evaluate** on all three difficulties — shared seeds, so every model sees the same worlds.
Smoke-test the connection first (`scripts/smoke_llm.sh meta-llama/Llama-3.2-1B-Instruct --base-url http://localhost:8000/v1 --steps 5 --coord easy`), then:

```bash
# Local open weights (vLLM on :8000)
scripts/run_llm_eval.sh meta-llama/Llama-3.2-1B-Instruct --base-url http://localhost:8000/v1 --episodes 20 --difficulty easy,medium,hard

# Hosted OpenAI
export OPENAI_API_KEY=sk-...
scripts/run_llm_eval.sh gpt-4o-mini --client openai --episodes 20 --difficulty easy,medium,hard

# Hosted Anthropic / Gemini — same shape, swap the client and key
export ANTHROPIC_API_KEY=sk-ant-...
scripts/run_llm_eval.sh claude-sonnet-4-20250514 --client anthropic --episodes 20 --difficulty easy,medium,hard
```

**3. Build the submission** (use the same model id you evaluated):

```bash
python scripts/make_submission.py --model-id meta-llama/Llama-3.2-1B-Instruct --name "Llama-3.2-1B-Instruct" --type open-weight --family Llama --params 1B
```

This reads the eval outputs and prints the ready-to-paste leaderboard entry — Base% / Coord.%
/ Total%, each with a 95% CI — and writes `outputs/submissions/<id>.zip` containing the
gameplay videos and debug traces we use to mark an entry **✓ verified**.

## MARL track

MARL agents train and evaluate on the same difficulty — one run each, on a separate
leaderboard track (symbolic and text are [not comparable](README.md#rl-vs-llm-interfaces)):

```bash
uv run --extra baselines-rl --python 3.12 python baselines/ippo_rnn.py \
  TRAINING_COORDINATION_DIFFICULTY=hard EVAL_DIFFICULTIES=[hard]
```

Repeat for `easy` and `medium`.

## Send it

Open a PR **against this repo** adding the printed entry to
[`data/leaderboard.json`](data/leaderboard.json) — the `homogeneous` list for LLM teams, or
the `marl` list for the MARL track — and attach the `.zip`. Prefer not to PR? Email both to
<k.tessera@ed.ac.uk>. The videos let us re-check the run and mark it ✓ verified. The
[website](https://alem-world.github.io/leaderboard.html) renders from this file.

## Standard submission (defaults)

Use these unless you state a deviation:

- **Harness** — `robust_all` (CoT, communication, scratchpad, reasoning all on).
- **Team** — zero-shot, homogeneous, 3 agents.
- **Difficulties** — `easy`, `medium`, `hard`, reported **separately** (never averaged).
- **Episodes** — 20 per difficulty (≥ 10 if cost-constrained), shared eval seeds (`EVAL_SEED=9999`).
- **Metrics** — Base%, Coord.%, Total%, each with a 95% CI — all produced by `make_submission.py`.

Change the harness, prompt mode, history, communication, scratchpad, or parsing? Say so — it's
part of the submission.
