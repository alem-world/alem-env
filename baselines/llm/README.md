# LLM Agent Baseline for ALEM

This module contains LLM evaluation runners, agents, clients, configs, and result
utilities. The reusable text environment interface now lives in `alem.llm` so it
can be used by this baseline code, external LLM repos, and human/debug tooling
without importing from `baselines`.

## Layout

```
baselines/llm/
  eval_alem.py                 # Main entry point (Hydra + W&B)
  alem_language_wrapper.py     # Compatibility shim -> alem.llm.alem_language_wrapper
  alem_language_wrapper_single.py
  alem_env.py                  # Compatibility shim -> alem.llm.alem_env
  ascii_map.py                 # Compatibility shim -> alem.llm.ascii_map
  play_human.py                # Pygame human-play mode for the language wrapper
  utils.py                     # Result collection, summary stats, seeding
  config/
    config.yaml                # Default Hydra configuration
  eval_utils/                  # BALROG-derived evaluation infrastructure
    __init__.py
    evaluator.py               # Multi-episode evaluation loop
    client.py                  # LLM API clients (vLLM, OpenAI, Anthropic, Gemini, …)
    prompt_builder.py          # Builds message history for the LLM (text + optional images)
    debug_visualiser.py        # Generates self-contained HTML debug viewer from JSONL
    agents/
      __init__.py              # AgentFactory — creates agents from config
      base.py                  # BaseAgent interface
      naive.py                 # Single-shot: observation → action
      chain_of_thought.py      # Think step-by-step, then ACTION: <action>
      robust_naive.py          # Uses <|ACTION|>...<|END|> structured output tags
      robust_cot.py            # CoT + structured output tags
      custom.py                # Maintains and updates an explicit plan each step
      few_shot.py              # Prepends demonstration episodes as in-context examples
      random.py                # Uniform random over all canonical action labels (no LLM)
      dummy.py                 # Always returns "wait" (debugging)
```

Canonical interface imports for new code:

```python
from alem.llm.alem_language_wrapper import AlemLanguageWrapper, make_alem_env
from alem.llm.alem_env import AlemTextEnv
from alem.llm.ascii_map import render_ascii_map
```

## Observation pipeline

```
EnvState (JAX arrays)
  │
  ▼
AlemLanguageWrapper.process_obs()
  ├── describe_env()        → visible blocks around the player
  ├── describe_status()     → health, role, dungeon level, sleeping/resting
  ├── describe_teammates()  → positions, roles, health, active requests
  ├── describe_inventory()  → vitals, attributes, items, tools, enchantments
  └── render_alem_pixels()  → PIL image (optional, for VLM agents)
  │
  ▼
{
  "text": {
    "long_term_context": "You see: tree 2 steps north, stone 5 steps east...",
    "short_term_context": "Your status: health 9/9, food 8/9... Your inventory: wood: 3..."
  },
  "image": <PIL.Image>   # pixel rendering of the agent's view
}
```

## Action space

The wrapper exposes 55 canonical Alem-Coop action labels as human-readable strings
(Move/Do/Place/Make/Drink/Cast/Read/Enchant/LevelUp/Request/Give/Build/...). The LLM emits
one of these strings; the wrapper maps it to the corresponding `Action` enum
index via case-insensitive fuzzy matching. Targeted transfers such as
`Give to Agent 2` are parsed into agent-specific Give slots. Invalid actions
default to `Noop`.

## Agent types

| Agent          | Description                                       |
|----------------|---------------------------------------------------|
| `robust_all`   | **Default — used for the paper experiments.** Structured `<action>` tags + CoT + communication + scratchpad memory, with multi-strategy fallback parsing and retry-on-failure. |
| `naive`        | Directly asks for an action, no reasoning         |
| `cot`          | Chain-of-thought, then `ACTION: <action>`         |
| `robust_naive` | Requires `<\|ACTION\|>...<\|END\|>` tags          |
| `robust_cot`   | CoT + structured output tags                      |
| `custom`       | Maintains an explicit plan, updated each step     |
| `few_shot`     | Prepends demonstration episodes as ICL examples   |
| `random`       | Uniform random over actions (no LLM call)         |
| `dummy`        | Always "wait" (debugging)                         |

## Prompt mode

`agent.prompt_mode` controls how much game knowledge and coordination information
each agent receives. This is the main knob for the experiments:

| Mode                     | What the agent gets                                                         |
|--------------------------|-----------------------------------------------------------------------------|
| `general`                | Action list + achievements only (no game rules or coordination info)        |
| `specific`               | Full game rules (survival, roles, crafting, progression); no coordination   |
| `specific_collaborative` | **Default.** Full game rules + coordination section + per-turn coord cues   |

> The older `wrapper.llm_mode` flag is **deprecated** — use `agent.prompt_mode` instead.

## Quick start

All quick-start commands below use the tested 3-agent ALEM setup. The single-agent
environment is experimental and should not be the default smoke-test path.

### 1) Text interface preview

```bash
python examples/llm_text_smoke.py --coord easy --show-affordances
```

This prints the system prompt excerpt, one text observation per agent, and a few
action parser examples without Hydra, W&B, or model calls.

### 2) Random agent (no LLM server)

```bash
WANDB_MODE=disabled JAX_PLATFORM_NAME=cpu python baselines/llm/eval_alem.py \
    agent.type=random \
    eval.num_episodes.alem=1 \
    eval.max_steps_per_episode=1 \
    eval.num_workers=1 \
    eval.save_images=false \
    eval.debug=false \
    eval.generate_debriefs=false
```

This is the fastest way to confirm that the environment, text wrapper, evaluator,
and output logging work before spending tokens on an API model. The first run
compiles JAX and can take around 30 seconds even for a one-step smoke test.

### 3) OpenAI smoke test

```bash
export OPENAI_API_KEY=sk-...
WANDB_MODE=disabled python baselines/llm/eval_alem.py \
    agent.type=naive \
    clients.0.client_name=openai clients.1.client_name=openai clients.2.client_name=openai \
    clients.0.model_id=gpt-4o-mini clients.1.model_id=gpt-4o-mini clients.2.model_id=gpt-4o-mini \
    eval.num_episodes.alem=1 \
    eval.max_steps_per_episode=1 \
    eval.num_workers=1 \
    eval.save_images=false \
    eval.debug=false \
    eval.generate_debriefs=false \
    agent.max_image_history=0 \
    agent.max_text_history=4
```

### 4) vLLM-served open model

Install vLLM in a **separate** virtual env — it pins its own torch/CUDA build that
would clash with this repo's pinned `jax`. A long HTTP timeout avoids failures on
the large CUDA wheels:

```bash
uv venv --python 3.12 .venv-vllm
UV_HTTP_TIMEOUT=600 uv pip install --python .venv-vllm/bin/python vllm
```

Terminal 1 — serve the model (port 8000 matches the config's default `base_url`,
so no `base_url` override is needed below):

```bash
.venv-vllm/bin/vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --port 8000 --gpu-memory-utilization 0.7 --max-model-len 12288
```

Terminal 2 — run eval from this repo's main env (`clients.N` maps to agent N):

```bash
python baselines/llm/eval_alem.py \
    clients.0.client_name=vllm clients.1.client_name=vllm clients.2.client_name=vllm \
    clients.0.model_id=meta-llama/Llama-3.2-1B-Instruct \
    clients.1.model_id=meta-llama/Llama-3.2-1B-Instruct \
    clients.2.model_id=meta-llama/Llama-3.2-1B-Instruct \
    eval.num_episodes.alem=3
```

This uses the default `robust_all` agent and `specific_collaborative` prompt mode
— i.e. the paper setup. For HF gated models (e.g. Gemma):

```bash
export HF_TOKEN=hf_...
hf auth login --token $HF_TOKEN
```

### 5) OpenAI longer run

```bash
export OPENAI_API_KEY=sk-...
python baselines/llm/eval_alem.py \
    agent.type=cot \
    clients.0.client_name=openai clients.1.client_name=openai clients.2.client_name=openai \
    clients.0.model_id=gpt-4o-mini clients.1.model_id=gpt-4o-mini clients.2.model_id=gpt-4o-mini \
    eval.num_episodes.alem=1 \
    eval.max_steps_per_episode=200 \
    agent.max_image_history=0 agent.max_text_history=16
```

### 6) Anthropic Claude

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python baselines/llm/eval_alem.py \
    agent.type=robust_cot \
    clients.0.client_name=anthropic clients.1.client_name=anthropic clients.2.client_name=anthropic \
    clients.0.model_id=claude-sonnet-4-20250514 \
    clients.1.model_id=claude-sonnet-4-20250514 \
    clients.2.model_id=claude-sonnet-4-20250514 \
    eval.num_episodes.alem=1 \
    eval.max_steps_per_episode=200 \
    agent.max_image_history=0 agent.max_text_history=16
```

## Configuration

All settings live in `config/config.yaml` and can be overridden via Hydra CLI.
Key options:

### Environment

```yaml
alem:
  num_agents: 3                    # 1-4 agents (3 = tested default)
  max_timesteps: 10000             # Max env timesteps before episode ends
  god_mode: false                  # Invincibility (debugging)
  coordination_difficulty: easy    # "none" | "easy" | "medium" | "hard"
  soft_specialization: true        # Role-based efficiency bonuses
  shared_reward: false             # Share rewards across agents

  wrapper:
    egocentric: false              # true = directions relative to agent facing
    render_pixel_size: 64          # 64 = paper-quality GIFs; 10 = low-mem
    render_downscale: 2            # use 2 when render_pixel_size=64
    use_ascii: false               # true = 9x11 ASCII grid instead of the "You see:" list
```

### Agent

```yaml
agent:
  type: "robust_all"               # see Agent types table
  prompt_mode: "specific_collaborative"  # see Prompt mode table
  use_cot: true
  use_communication: true          # free-form broadcast to teammates each step
  use_scratchpad: true             # private per-agent memory across steps
  max_text_history: 8
  max_image_history: 0             # 0 = text-only; >=1 = include pixel renderings (VLM)
  max_cot_history: 1
```

### Clients (LLM API)

```yaml
clients:
  - client_name: "vllm"            # "vllm" | "openai" | "anthropic" | "gemini" | "nvidia" | "xai"
    model_id: "google/gemma-4-E2B-it"
    base_url: "http://localhost:8000/v1"
    generate_kwargs:
      temperature: 1.0
      max_tokens: 8192
    timeout: 420
    max_retries: 5
```

`clients.i` maps to `agent_id=i`. Provide one entry per agent.

### Evaluation

```yaml
eval:
  num_episodes:
    alem: 20
  max_steps_per_episode: 10000
  num_workers: 4                   # parallel eval workers; raise if you have resources
  feedback_on_invalid_action: true
  save_images: true                # GIFs + debug JSONL/HTML
  generate_debriefs: true          # post-episode LLM debriefs (expensive for API models)
```

## Outputs

Each run creates a timestamped directory under `outputs/alem_eval/` containing
per-episode CSVs, episode-stat JSON, a debug JSONL plus a self-contained HTML
debug viewer, and (when `save_images=true`) per-agent and combined GIFs.

Results are also logged to W&B unless `WANDB_MODE=disabled` is set.

## W&B metrics

LLM eval logs under the same `eval/` namespace as the RL `run_final_eval`,
enabling direct comparison in W&B:

| Key                                          | Description                                                       |
|----------------------------------------------|-------------------------------------------------------------------|
| `eval/Team/achievement_pct`                  | Fraction of achievements completed by any agent (team total)      |
| `eval/Team/coordination_achievement_pct`     | Fraction of coordination-specific achievements                    |
| `eval/Team/normal_achievement_pct`           | Fraction of non-coordination achievements                         |
| `eval/Agent{N}/achievement_pct`              | Per-agent achievement fraction                                    |
| `eval/Achievements/{name}`                   | Per-achievement rate across episodes (0-100)                      |
| `eval/Cooperation/trade_count`               | Number of resource transfers between agents                       |
| `eval/Cooperation/revives`                   | Number of agent revives                                           |
| `eval/Coordination/handover_completion_rate` | Handover success rate                                             |
| `eval/episode_returns`                       | Mean reward per agent per episode                                 |
| `eval/episode_lengths`                       | Mean steps per episode                                            |
| `eval/num_episodes`                          | Number of episodes evaluated                                      |
| `eval/action_parse_rate`                     | Fraction of LLM outputs successfully parsed                       |

All metrics are computed via `compute_score()` — the same function used by RL
training — so RL/LLM scores are directly comparable. For best comparability with
RL, use `eval.max_steps_per_episode=10000`.

## Adding a new agent

1. Create `eval_utils/agents/my_agent.py` with a class extending `BaseAgent`.
2. Implement `act(self, obs, prev_action=None)` returning an `LLMResponse`.
3. Register it in `eval_utils/agents/__init__.py` (`AgentFactory.create_agent`).
4. Use it: `python baselines/llm/eval_alem.py agent.type=my_agent`.
