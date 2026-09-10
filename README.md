<h1 align="center">
  <a href="https://arxiv.org/abs/2606.08340">
    <img src="images/alem-logo-animated.gif" alt="Alem logo showing synchronous coordination" width="360" />
  </a>
</h1>

<p align="center">
  <a href="https://pypi.org/project/alem-env/"><img alt="PyPI" src="https://img.shields.io/pypi/v/alem-env.svg" /></a>
  <a href="https://pypi.org/project/alem-env/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/alem-env.svg" /></a>
  <a href="https://github.com/alem-world/alem-env/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/alem-world/alem-env/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" /></a>
  <a href="https://arxiv.org/abs/2606.08340"><img alt="arXiv:2606.08340" src="https://img.shields.io/badge/arXiv-2606.08340-b31b1b.svg" /></a>
  <a href="https://huggingface.co/alem-world/alem-rl-baselines"><img alt="Hugging Face Models" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffce1c.svg" /></a>
  <a href="https://alem-world.github.io/"><img alt="Website" src="https://img.shields.io/badge/website-alem--world.github.io-111111.svg" /></a>
</p>

<h3 align="center">Can LLM agents coordinate in long-horizon, open-ended tasks?</h3>

<p align="center">
  <a href="https://alem-world.github.io/">🌐 Website</a> · <b><a href="https://alem-world.github.io/leaderboard.html">🏆 Leaderboard</a></b> · <b><a href="#evaluate-an-llm">⚡ Evaluate an LLM</a></b> · <a href="SUBMISSION.md">Submit a result</a> · <a href="https://arxiv.org/abs/2606.08340">📄 Paper</a> · <a href="https://huggingface.co/alem-world/alem-rl-baselines">🤗 Models</a>
</p>

*Alem* is a JAX benchmark for open-ended multi-agent coordination. Building on [Craftax](https://github.com/MichaelTMatthews/Craftax) and [Multi-Agent Craftax / Craftax-Coop](https://github.com/BaselOmari/MA-Craftax), *Alem* introduces procedurally generated coordination tasks, soft specialisation, communication, and controllable coordination difficulty into a long-horizon survival world with exploration, crafting, trading, and combat. The same world is exposed through symbolic, pixel, and text interfaces, making it usable by MARL agents, language agents, and humans.

*Alem* means *world* in Amharic.

<p align="center"><sub><b>93</b> achievements&nbsp;·&nbsp;<b>27</b> coordination goals&nbsp;·&nbsp;<b>3</b> difficulty tiers&nbsp;·&nbsp;<b>9</b> dungeon levels&nbsp;·&nbsp;episodes up to <b>10,000</b> steps&nbsp;·&nbsp;pure JAX</sub></p>

## Contents

[RL Playing](#rl-agents-playing) · [LLM Playing](#llm-agents-playing) · [Install](#install) · [Quick Start](#quick-start) · [**Evaluate an LLM**](#evaluate-an-llm) · [Configure](#configure) · [RL Agents](#rl-agents) · [Baselines](#baselines) · [Human Play](#human-play) · [Docker](#docker) · [Package Layout](#package-layout) · [Development](#development) · [RL vs LLM Interfaces](#rl-vs-llm-interfaces) · [Reproduce the Paper](#reproduce-the-paper) · [Submit to the Leaderboard](#submit-to-the-leaderboard) · [Errata](#errata) · [Contributing](#contributing) · [Citation](#citation) · [License](#license)

## RL Agents Playing

A team of MARL agents controlling the three players from symbolic observations, each acting from its own egocentric view.

<p align="center">
  <img src="images/sample_agents_playing.gif" alt="RL agents playing Alem" width="760" />
</p>

Fast to train end-to-end in JAX. Full MARL training code and reference baselines live in [`baselines/`](baselines); see [Baselines](#baselines).

## LLM Agents Playing

The same world through the text interface. Each agent gets its own observation, broadcasts a free-form message to teammates every step and stores important information in scratchpad memory.

<p align="center">
  <img src="images/llm_agents_playing.gif" alt="LLM agents reasoning and coordinating in Alem" width="820" />
</p>

<p align="center"><sub>Gemini 3.1 Pro (medium). <b>THINKING</b> = the agent's private plan; <b>MESSAGE</b> = what it broadcasts to the team. The view follows whichever agent is speaking; reasoning is typed out for readability, not the agents' real decision speed.</sub></p>

Three agents negotiate a synchronous action. They plan ahead, model each other, lock in a shared step, and then execute it together:

- **Plans ahead.** Each agent commits to a turn-indexed plan (`Step 5 Move · Step 6–7 Noop · Step 8 DO`) and waits in position.
- **Theory of mind.** A0 predicts a teammate's move will break the plan (*"A2, your Step-6 DO will fail because A1 changed to Step 8"*), and A1 catches the clash: *"A0 and A2 sent conflicting times — EVERYONE DO ON STEP 8!"*
- **Coordinates out loud.** All three line up and DO the 3-agent tree on the same step.

See [Evaluate an LLM](#evaluate-an-llm) to run it yourself.

<details>
<summary><b>👁️ Click to see what the agents actually see</b></summary>

<br>

Every step a language agent gets a **system prompt** (the rules, sent once) and a **text observation** (its current view), and must reply with an `<action>`, an optional `<communication>` broadcast, and an optional private `<scratchpad>`. We use **progressive disclosure**, where we only give relevant information for the current level in the prompt, and add information as agents get to more levels.

**System prompt template.** Placeholders in `{…}` are filled per agent/run (abridged; the full rules are sent verbatim):

```text
You are Agent {id} ({role}) in a {num_agents}-agent cooperative survival game. Your goal is to gather resources, craft gear, fight monsters, and descend through {num_levels} dungeon levels, while coordinating with teammates. You must survive — if your health reaches zero, you die, and if all agents die the game ends. Maximize achievements while alive.

[ … full game rules: movement & facing, Do/interaction, crafting recipes, roles & specialist penalties, coordination (sync / handover / construction), the resource chain, and the {num_achievements} achievements …]

<output_format>
1. (Required) Exactly one action from the available list:
   <action>YOUR_CHOSEN_ACTION</action>
2. (Optional) Broadcast to teammates, up to {comm_char_limit} chars:
   <communication>YOUR_MESSAGE</communication>
3. (Optional) Private notes, up to {scratchpad_char_limit} chars — not shared; your only memory:
   <scratchpad>YOUR_NOTES</scratchpad>
Token budget: {token_budget} tokens for the full response (including reasoning).
</output_format>
```

**[View the full system prompt, filled in](SYSTEM_PROMPT.md)** (a concrete 3-agent example on overworld).

**Observation template.** The structure every agent receives each step:

```text
Step: {step}/{max_steps} ({steps_remaining} remaining, ends early if all agents die)
Position: (x={x}, y={y})
Role: {role}
Location: {dungeon_level}
Achievements: {unlocked}/{total} ({locked} unlock later)

You see:
- {object} {relative_position} (x={x}, y={y})
  ...one line per visible object...

Facing: {direction}. Do target: {object_in_front} (x={x}, y={y}).

Coordination:
- {object} (x={x}, y={y}): {how_this_object_must_be_coordinated}
  ...one line per coordination-relevant object...

Teammates:
Agent {id} ({role}): {relative_position} (x={x}, y={y}), health={hp}
  ...one line per teammate...

Your status: health {hp}, food {food}, drink {drink}, energy {energy}, mana {mana}, xp {xp}
Available actions: {legal_actions_this_step}
```

**Filled-in example.** What the warrior actually sees at step 0:

```text
Step: 0/10000 (10000 remaining, ends early if all agents die)
Position: (x=24, y=24)
Role: warrior
Location: Overworld (surface)
Achievements: 0/93 (39 unlock later)

You see:
- stone 5 steps east (x=29, y=24)
- tree 1 step north and 2 steps west (x=22, y=23)
- construction_site 2 steps north (x=24, y=22)
- iron 3 steps south and 5 steps east (x=29, y=27)

Facing: north. Do target: grass (x=24, y=23).

Coordination:
- construction_site (x=24, y=22): requires 3 agents to Build simultaneously (fails alone).
- tree (x=26, y=23): one agent begins, another completes it within 6 steps (handover).
- stone (x=20, y=21): works solo, but a bonus when 3 agents Do simultaneously.

Teammates:
Agent 1 (forager): 1 step east (x=25, y=24), health=9
Agent 2 (miner): 1 step south (x=24, y=25), health=9

Your status: health 9, food 9, drink 9, energy 9, mana 9, xp 0
Available actions: Noop, Move {West,East,North,South}, Do, Sleep, Rest, Request {Food,Drink,Wood,Stone,Iron,Coal,Diamond,Ruby,Sapphire}
```

</details>

## Install

```bash
uv venv --python 3.12
uv pip install alem-env # latest release from PyPI     
```

Or from source for development (editable install):

```bash
uv pip install -e .
```

Optional extras (work with either `alem-env` or `-e .`):

```bash
uv pip install -e ".[llm]"             # OpenAI-compatible LLM interface
uv pip install -e ".[play]"            # pygame human-play example
uv pip install -e ".[gpu]"             # NVIDIA CUDA 12 JAX wheels
uv pip install -e ".[baselines-rl]"    # JAX MARL trainers (see Baselines)
uv pip install -e ".[baselines-llm]"   # LLM-agent evaluation harness
uv pip install -e ".[dev]"             # pytest, ruff, jaxtyping
```

Plain `pip` also works (no uv required):

```bash
pip install -e .
pip install -e ".[gpu]"
```

Or `pip install alem-env`.
## Quick Start

```python
import jax
from alem.alem_env import make_alem_env_from_name

env = make_alem_env_from_name("Alem-Coop-Symbolic")
obs, state = env.reset(jax.random.PRNGKey(0))

rng_act = jax.random.split(jax.random.PRNGKey(1), env.num_agents)
actions = {agent: env.action_space(agent).sample(rng_act[i]) for i, agent in enumerate(env.agents)}

obs, state, rewards, dones, infos = env.step(jax.random.PRNGKey(2), state, actions)
```

Available environments:

| Name                        | Description                                              |
| --------------------------- | -------------------------------------------------------- |
| `Alem-Coop-Symbolic`        | Full multi-agent environment, symbolic observations      |
| `Alem-Coop-Pixels`          | Full multi-agent environment, pixel observations         |
| `Alem-Coop-Symbolic-Debug`  | Smaller debug environment, only overworld (first floor). |
| `Alem-SingleAgent-Symbolic` | Single-agent variant (experimental)                      |

## Evaluate an LLM

Alem is an **open leaderboard for multi-agent coordination**: evaluate any OpenAI-compatible model and [submit your score](https://alem-world.github.io/leaderboard.html). Three agents play as a team through the text interface: each reasons privately, broadcasts a message to teammates, and keeps a scratchpad.

```bash
uv pip install -e ".[baselines-llm]"   # the LLM evaluation harness
```

**Local model** (vLLM or any OpenAI-compatible server). The same `MODEL_ID` drives all three agents:

```bash
# Install vLLM in a SEPARATE env; it pins its own torch/CUDA build that clashes with this repo's jax
uv venv --python 3.12 .venv-vllm
source ~/.venv-vllm/bin/activate
uv pip install vllm --torch-backend=auto

# Terminal 1: serve your model
vllm serve meta-llama/Llama-3.2-1B-Instruct --port 8000 

# Terminal 2: evaluate a 3-agent team on all leaderboard difficulties (from this repo's main env)
scripts/run_llm_eval.sh meta-llama/Llama-3.2-1B-Instruct \
    --base-url http://localhost:8000/v1 --episodes 20 --difficulty easy,medium,hard
```

> Any OpenAI-compatible server works; vLLM is just the common choice for open models. See [vLLM install docs](https://docs.vllm.ai/en/stable/getting_started/installation.html) for GPU/CPU build options.

**Hosted API** (OpenAI / Anthropic / Gemini / …):

```bash
export OPENAI_API_KEY=sk-...
python baselines/llm/eval_alem.py \
    clients.0.client_name=openai clients.1.client_name=openai clients.2.client_name=openai \
    clients.0.model_id=gpt-4o-mini clients.1.model_id=gpt-4o-mini clients.2.model_id=gpt-4o-mini \
    eval.num_episodes.alem=20
```

Swap `client_name` to `anthropic`, `gemini`, `nvidia`, or `xai` (and set the matching API key). Runs use the default `robust_all` agent + `specific_collaborative` prompt, the paper setup.

**Try it before spending tokens.** Run a cheap 5-step smoke test, or preview the text observations with no model at all:

```bash
scripts/smoke_llm.sh meta-llama/Llama-3.2-1B-Instruct --base-url http://localhost:8000/v1 --steps 5 --coord easy
uv run python examples/llm_text_smoke.py --coord easy --show-affordances   # no model calls
```

**Prefer Docker?** A single `docker run` serves your model with vLLM *and* runs the 3-agent eval, with no local install. See [Docker → Evaluate an LLM](#evaluate-an-llm-in-one-command).

→ **Full harness reference**: agent types, prompt modes, providers, per-agent configs, and metrics: [`baselines/llm/README.md`](baselines/llm/README.md). To publish a result: [`SUBMISSION.md`](SUBMISSION.md).

## Configure

```python
from alem.alem_coop.alem_state import EnvParams, StaticEnvParams, get_coordination_params

env_params = EnvParams().replace(
    **get_coordination_params("easy"),
    soft_specialization=True,
    shared_reward=False,
)
static_env_params = StaticEnvParams(player_count=3, num_comm_channels=4)

env = make_alem_env_from_name(
    "Alem-Coop-Symbolic",
    env_params=env_params,
    static_env_params=static_env_params,
)
```

Coordination difficulty can be `"none"`, `"easy"`, `"medium"`, `"hard"`, or a numeric alpha in `[0, 1]`.

## RL Agents

A minimal framework-free rollout using symbolic observations and legal action masks:

```bash
uv run python examples/random_rl_agent.py --coord easy --steps 100
uv run python examples/random_rl_agent.py --players 2 --coord hard --steps 200
```

The example uses a jitted `lax.scan` loop and can serve as a template for custom policies. Full MARL training recipes (IPPO, HyperMARL-IPPO, MAPPO, PQN-VDN) live in [`baselines/`](baselines); see [Baselines](#baselines).

## Baselines

Reference MARL training code and the LLM-agent evaluation harness live under [`baselines/`](baselines). Each RL algorithm is a single self-contained file (CleanRL-style) with a matching [Hydra](https://hydra.cc) config; the LLM harness (derived from [BALROG](https://github.com/balrog-ai/BALROG)) drives 3 language agents through the text interface.

```bash
uv pip install -e ".[baselines-rl]"    # JAX MARL trainers (IPPO / MAPPO / PQN-VDN / HyperMARL)
uv pip install -e ".[baselines-llm]"   # LLM-agent evaluation harness
```

| Guide | What's inside |
| ----- | ------------- |
| [**RL baselines →** `baselines/README.md`](baselines/README.md) | IPPO / HyperMARL-IPPO / MAPPO / PQN-VDN trainers, run commands, and reloading the [pretrained checkpoints](https://huggingface.co/alem-world/alem-rl-baselines) from the paper. |
| [**LLM harness →** `baselines/llm/README.md`](baselines/llm/README.md) | Agent types, prompt modes, providers (vLLM / OpenAI / Anthropic / Gemini), configuration, and metrics. |

```bash
cd baselines && python ippo_rnn.py TOTAL_TIMESTEPS=10000           # train an RL policy
cd baselines/llm && python eval_alem.py clients.0.client_name=openai \
    clients.1.client_name=openai clients.2.client_name=openai      # evaluate an LLM team
```

## Human Play

Install the optional play dependencies first:

```bash
uv pip install -e ".[play]"
```

```bash
uv run python examples/play_alem.py
uv run python examples/play_alem.py --players 3 --coord easy --seed 42
uv run python examples/play_alem.py --players 2 --coord hard --god
```

| Key         | Action           |
| ----------- | ---------------- |
| `W A S D`   | Move             |
| `Space`     | Do / interact    |
| `Tab`       | Sleep            |
| `E`         | Rest             |
| `.` / `,`   | Descend / ascend |
| `Backspace` | Give to teammate |
| `Q`         | No-op            |

The game advances after all players have chosen an action.

## Docker

Three images, each built from the repo root. Two of them run a full benchmark job in a **single `docker run`**; the container installs nothing and starts working immediately:

| Image | Dockerfile | What one `docker run` does |
| ----- | ---------- | -------------------------- |
| `alem-llm` | `docker/Dockerfile.llm` | Serves your model with vLLM **and** runs a 3-agent LLM evaluation against it |
| `alem-rl`  | `docker/Dockerfile.rl`  | Runs a JAX MARL trainer (IPPO / MAPPO / PQN-VDN / HyperMARL) |
| `alem-env` | `docker/Dockerfile.env` | Lightweight environment-only image for the examples / your own code |

Both `alem-llm` and `alem-rl` need [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host and `--gpus` at runtime.

### Evaluate an LLM in one command

```bash
docker build -f docker/Dockerfile.llm -t alem-llm .

# Serves the model with vLLM, waits until ready, then evaluates a 3-agent team.
# Only MODEL_ID is required; mount the HF cache so models are downloaded once.
docker run --rm --gpus all --shm-size=16g \
    -e MODEL_ID=meta-llama/Llama-3.2-1B-Instruct \
    -e HF_TOKEN=$HF_TOKEN \
    -v ~/.cache/huggingface:/app/.cache/huggingface \
    alem-llm eval.num_episodes.alem=5
```

Trailing arguments are Hydra overrides for the eval (e.g. `eval.num_episodes.alem=5`, `alem.coordination_difficulty=hard`). For a large model, shard across GPUs with `-e VLLM_EXTRA_ARGS="--tensor-parallel-size 4"` and a bigger `--shm-size`. To log to W&B, pass `-e WANDB_API_KEY=... -e WANDB_MODE=online`. Full knobs are in the header of [`docker/Dockerfile.llm`](docker/Dockerfile.llm).

### Train an RL policy in one command

```bash
docker build -f docker/Dockerfile.rl -t alem-rl .

docker run --rm --gpus all -v "$PWD/outputs:/app/outputs" \
    alem-rl ippo_rnn TOTAL_TIMESTEPS=1e4 NUM_ENVS=16 SEED=0
```

The first argument is the trainer (`ippo_rnn`, `ippo_rnn_nops`, `ippo_hypermarl_rnn`, `mappo_rnn`, `pqn_vdn_rnn`); the rest are Hydra overrides. Run with no arguments to list the available trainers. Checkpoints and GIFs land in the mounted `outputs/`.

### Environment-only image

```bash
# CPU (default)
docker build -f docker/Dockerfile.env -t alem-env .

# GPU: NVIDIA CUDA 12
docker build -f docker/Dockerfile.env --build-arg ALEM_ACCELERATOR=cuda12 -t alem-env:gpu .
```

The image uses the system Python (`UV_SYSTEM_PYTHON=1`), so inside a container you call `python` directly, no `uv run` prefix needed.

```bash
docker run --rm alem-env                                           # smoke test (default CMD)
docker run --rm --gpus all alem-env:gpu                            # GPU smoke test
docker run --rm alem-env python examples/random_rl_agent.py --steps 20
docker run --rm alem-env python examples/llm_text_smoke.py --coord easy
docker run --rm -it alem-env bash                                  # interactive shell
```

> **Human play is easiest natively**: `uv pip install -e ".[play]"` then `uv run python examples/play_alem.py`. Pygame opens a real window with no display plumbing.

<details>
<summary><b>Running human play inside Docker (X11 setup)</b></summary>

<br>

Inside Docker, human play needs an X11 display *and* an image built with the `play` extra (so the SDL/X11 libs are present):

```bash
# Build with the play extra (adds pygame + SDL/X11 runtime libs)
docker build -f docker/Dockerfile.env --build-arg ALEM_EXTRAS=play -t alem-env:play .
```

- **Linux:** grant the container access to your X server first (this is the step that's usually missing when the window never appears), then run it:
  ```bash
  xhost +local:root
  docker run --rm -it --network host -e DISPLAY=$DISPLAY \
      -v /tmp/.X11-unix:/tmp/.X11-unix alem-env:play python examples/play_alem.py
  xhost -local:root   # revoke when done
  ```
- **macOS / Windows:** start XQuartz (macOS) or VcXsrv (Windows), enable "allow connections from network clients", then set `DISPLAY` accordingly.

</details>

## Package Layout

<details>
<summary><b>Repository map: where each piece lives</b></summary>

<br>

| Path                           | Purpose                                                               |
| ------------------------------ | --------------------------------------------------------------------- |
| `alem/alem_env.py`             | Environment factory                                                   |
| `alem/alem_coop/envs/`         | Symbolic, pixel, debug, and single-agent env classes                  |
| `alem/alem_coop/alem_state.py` | State dataclasses and `EnvParams` / `StaticEnvParams`                 |
| `alem/alem_coop/game_logic.py` | Step logic: movement, resources, combat, crafting, coordination       |
| `alem/alem_coop/world_gen/`    | Procedural world generation                                           |
| `alem/alem_coop/renderer/`     | Symbolic, pixel, and text renderers                                   |
| `alem/alem_coop/constants.py`  | Actions, achievements, blocks, items, textures                        |
| `alem/llm/`                    | Text observations, action parsing, ASCII maps, LLM evaluator adapters |
| `examples/random_rl_agent.py`  | Masked-random symbolic rollout (RL template)                          |
| `examples/llm_text_smoke.py`   | Preview text observations without model calls                         |
| `examples/llm_openai_smoke.py` | One-step OpenAI-compatible LLM smoke test                             |
| `examples/play_alem.py`        | Human pygame player                                                   |

</details>

## Development

```bash
uv pip install -e ".[dev]"   # pytest, ruff, jaxtyping
uv run pytest alem/tests/    # run the test suite
```

<details>
<summary><b>Lint &amp; format (ruff)</b></summary>

<br>

Code style is enforced with [ruff](https://docs.astral.sh/ruff/) (config in `pyproject.toml`). CI runs these checks before the test suite, so run them locally first:

```bash
uv run ruff check .          # lint
uv run ruff format --check . # verify formatting
```

To auto-fix before committing:

```bash
uv run ruff check --fix .    # apply safe lint fixes
uv run ruff format .         # format the code
```

</details>

## RL vs LLM Interfaces

Both interfaces drive the **same** environment but are **not directly comparable**; treat cross-paradigm scores as indicative, not head-to-head.

|                     | MARL (symbolic)                                          | LLM (text)                                                                        |
| ------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Observation**     | Numeric vector                                           | Natural-language text                                                             |
| **Communication**   | A discrete signal on one of `num_comm_channels` (e.g. 4) | Free-form text, ≤ 400 chars                                                       |
| **Comms vs acting** | Costs your action that turn                              | Sent alongside the action                                                         |
| **Memory**          | Recurrent hidden state                                   | Private `<scratchpad>` notes: not shared; the agent's only memory across steps.  |
| **Learning**        | Trained from reward                                      | Zero-shot                                                                         |

(`Request`/`Give` resource transfers are ordinary actions in both.)

Text observations apply lightweight preprocessing, including compact local-state summaries (inspired by [BALROG](https://github.com/balrog-ai/BALROG/blob/b7afe79e3e4265811cfa985ed7c95c4d1a11e3f5/balrog/environments/crafter/env.py#L129)) and action affordances. See the [language wrapper](alem/llm/alem_language_wrapper.py) for details.

## Reproduce the Paper

The paper's full experiments (the 13-LLM evaluation and the IPPO, HyperMARL-IPPO, MAPPO, PQN-VDN baselines) live in [`baselines/`](baselines); see [Baselines](#baselines) for launch commands and configs.

The paper's numbers were produced against *Alem* [`v0.1.0`](https://github.com/alem-world/alem-env/releases/tag/v0.1.0). For the exact settings to use when reporting an Alem number (seeds, episode count, metrics), see the canonical [evaluation protocol](EVALUATION.md).

## Submit to the Leaderboard

Alem is an open benchmark. To add an LLM agent, MARL policy, or custom harness result to the [leaderboard](https://alem-world.github.io/leaderboard.html), follow [`SUBMISSION.md`](SUBMISSION.md).

If your model is behind an OpenAI-compatible endpoint such as vLLM, the LLM path is one command:

```bash
scripts/run_llm_eval.sh YOUR_MODEL_ID --base-url http://localhost:8000/v1 --episodes 20 --difficulty easy,medium,hard
```

Use `scripts/smoke_llm.sh YOUR_MODEL_ID --base-url http://localhost:8000/v1 --steps 5 --coord easy` first for a cheap smoke test. Submissions should include enough detail to reproduce the run: model or algorithm, harness, coordination difficulty, seeds, metrics, evaluation date, and any non-default config. Pull requests are preferred, but issues and email submissions are also accepted.

## Errata

Fixes that change environment behaviour are listed here, so numbers produced on different versions can be compared with the difference in mind. Report an Alem number together with the version it was produced on.

- **Prior to v0.2.0 - ladder placement.** The up-ladder and down-ladder positions were drawn independently and never checked against each other, so they could share a row and the up-ladders would replace some of the down-ladders. Affected levels could have fewer ways down than intended. This should be a minor issue for most runs, but it does change generated worlds, so results from before and after the fix are not directly comparable.

- **Prior to v0.2.1 - lava bridging.** Stone can now be placed into lava and mined into a walkable path, restoring Craftax's Fire Realm bridging mechanic without changing generated maps. This changes episode behaviour, but no agents have reached the Fire Realm yet, so it doesn't affect scores. 

## Contributing

Contributions are welcome: new baselines, bug fixes, docs, and coordination tasks. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev setup, lint/test workflow, and PR checklist, and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community expectations.

## Citation

```bibtex
@article{tessera2026alem,
  title   = {Benchmarking Open-Ended Multi-Agent Coordination in Language Agents},
  author  = {Tessera, {Kale-ab} Abebe and Szecsenyi, Andras and Barker, Cameron and
             Rutherford, Alexander and Paglieri, Davide and Scannell, Aidan and
             Gouk, Henry and Crowley, Elliot J. and Rockt\"{a}schel, Tim and
             Storkey, Amos},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.08340}
}
```

## License

MIT. See `LICENSE`.
