# Baselines

Reference implementations used for the *Alem* paper. Two families live here:

- **RL (MARL) trainers** — this directory (`baselines/*.py`), documented below.
- **LLM-agent evaluation harness** — [`baselines/llm/`](llm), documented in [`baselines/llm/README.md`](llm/README.md).

Following the [CleanRL](https://github.com/vwxyzjn/cleanrl) philosophy — and
[JaxMARL](https://github.com/FLAIROx/JaxMARL), which these are adapted from — each RL
algorithm is a single self-contained file with a matching [Hydra](https://hydra.cc)
config in [`config/`](config).

## Install

Install only the set you need (from the repo root):

```bash
uv pip install -e ".[baselines-rl]"    # JAX MARL trainers (IPPO / MAPPO / PQN-VDN / HyperMARL)
uv pip install -e ".[baselines-llm]"   # LLM-agent evaluation harness
```

## RL training

| Algorithm                       | Entry point                       | Reference                                              |
| ------------------------------- | --------------------------------- | ------------------------------------------------------ |
| IPPO (RNN, shared params)       | `ippo_rnn.py`                     | [IPPO](https://arxiv.org/abs/2011.09533)    |
| IPPO (RNN, no param sharing)    | `ippo_rnn_nops.py`                | [IPPO](https://arxiv.org/abs/2011.09533)    |
| HyperMARL-IPPO (RNN)            | `ippo_hypermarl_rnn.py`           | [HyperMARL](https://arxiv.org/abs/2412.04233) ([code](https://github.com/KaleabTessera/HyperMARL)) |
| MAPPO (RNN)                     | `mappo_rnn.py`                    | [MAPPO](https://arxiv.org/abs/2103.01955)    |
| PQN-VDN (RNN)                   | `pqn_vdn_rnn.py`                  | [PQN](https://arxiv.org/abs/2407.04811) ([code](https://github.com/mttga/purejaxql))  |

Run the trainers from this directory and override config values on the command line:

```bash
cd baselines
python ippo_rnn.py TOTAL_TIMESTEPS=10000
python mappo_rnn.py TRAINING_COORDINATION_DIFFICULTY=hard TOTAL_TIMESTEPS=10000  # override any config value
```

## Running stored policies

Pretrained RL checkpoints from the paper are on the Hugging Face Hub at
[**alem-world/alem-rl-baselines**](https://huggingface.co/alem-world/alem-rl-baselines):
120 checkpoints = 2 training budgets (`100M`, `1B` env steps) × 4 algorithms × 3
difficulties × 5 seeds, laid out as `<budget>/<algorithm>/<difficulty>/seed<N>/`.

Each trainer can skip training and instead restore a saved checkpoint, then run the
same final evaluation (and visualization) used after training. Download the checkpoints,
then pass `LOAD_CHECKPOINT` pointing at the checkpoint directory:

```bash
# 1. Download the checkpoints (needs: pip install -U huggingface_hub)
hf download alem-world/alem-rl-baselines --local-dir alem-rl-baselines

# 2. Reload and evaluate an IPPO policy (note NUM_COMM_CHANNELS=4)
cd baselines
python ippo_rnn.py \
    +LOAD_CHECKPOINT=../alem-rl-baselines/1B/ippo-rnn/hard/seed0/checkpoint \
    NUM_COMM_CHANNELS=4 \
    EVAL_DIFFICULTIES=[hard] \
    +VISUALIZE=True
```

Gifs are saved in `./outputs/`, set `VISUALIZE=False` to skip rendering and only run the numeric evaluation.

> **Important — the config must match how the checkpoint was trained.** Checkpoint
> shapes are fixed at training time, so the env config (number of agents, communication
> channels, etc.) must match or the restore will fail with a shape mismatch. The released
> checkpoints were all trained with **4 communication channels**, so load them with
> `NUM_COMM_CHANNELS=4`. The exact overrides for any checkpoint are stored under
> `reload_overrides` in its `config.json`.

## LLM-agent evaluation

The harness (derived from [BALROG](https://github.com/balrog-ai/BALROG)) drives 3
language agents through the text interface and supports vLLM, OpenAI, Anthropic, Gemini,
and other OpenAI-compatible providers. See [`llm/README.md`](llm/README.md) for the full
launch commands, agent types, prompt modes, and configuration.

```bash
cd baselines/llm
export OPENAI_API_KEY=sk-...
python eval_alem.py \
    clients.0.client_name=openai \
    clients.1.client_name=openai \
    clients.2.client_name=openai
```

## Reproduce the paper

The paper's numbers were produced against *Alem* [`v0.1.0`](https://github.com/alem-world/alem-env/releases/tag/v0.1.0).
For the exact settings to use when reporting an Alem number — seeds, episode count,
metrics — see the canonical [evaluation protocol](../EVALUATION.md).
