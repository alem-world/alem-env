# Alem Evaluation Protocol

This is the canonical protocol for reporting a number on *Alem*. RL and LLM
agents are scored by the **same** `compute_score()` function on the **same**
seeds, so results are reproducible and the leaderboard is comparable within each
track. Use these settings unless your paper explicitly states a deviation.

## Standard settings

| Setting | Value | Where it lives |
| --- | --- | --- |
| Environment | `Alem-Coop-Symbolic` | `ENV_NAME` |
| Agents | 3 | `alem.num_agents` |
| Soft specialisation | on | `alem.soft_specialization` |
| Shared reward | off | `alem.shared_reward` |
| Episodes | 20 per difficulty | `eval.num_episodes.alem` (LLM) / `TEST_NUM_EPISODES` (RL) |
| Max steps / episode | 10000 | `eval.max_steps_per_episode` / `TEST_MAX_STEPS` |
| Eval seed | 9999 | `EVAL_SEED` (shared by both tracks) |
| Coordination difficulty | `easy`, `medium`, `hard` | `coordination_difficulty` / `EVAL_DIFFICULTIES` |

**LLM agent (headline).** `agent.type=robust_all` with `prompt_mode=specific_collaborative`,
CoT, communication, and scratchpad all on (these are the config defaults). Set
`agent.reasoning=True` for models that emit a separate reasoning field (e.g. vLLM
with `--reasoning-parser`); leave it off for models that do not (e.g. GPT-4o).

**Seeding.** Episode `i` uses world seed `EVAL_SEED + i` (i.e. `9999 … 10018`).
This is identical for RL and LLM, so both see the same 20 worlds per difficulty.
Episodes end early when all agents die (typically well before the 10000-step cap).

**Coordination difficulty.** Evaluate on all three of `easy`, `medium`, and
`hard`, and report each *separately* — do not average across them. LLM agents are
swept over the three difficulties zero-shot; RL agents are trained **and**
evaluated on each difficulty (one model per difficulty). State the difficulty
next to every number.

## Headline metric

`eval/Team/achievement_pct` — the fraction of achievements unlocked by **any**
agent, averaged over the 20 episodes. Always report alongside it:

- `eval/Team/coordination_achievement_pct` — coordination-specific achievements
- `eval/Team/normal_achievement_pct` — non-coordination achievements
- `eval/action_parse_rate` (LLM only) — fraction of outputs successfully parsed;
  a low value means the score is throttled by formatting failures, not capability

The full metric list (per-agent, per-achievement, cooperation, coordination) is
documented in [`baselines/llm/README.md`](baselines/llm/README.md#wb-metrics).

## RL vs LLM

The symbolic (RL) and text (LLM) interfaces drive the same world but are **not
directly comparable** — see [RL vs LLM Interfaces](README.md#rl-vs-llm-interfaces).
Keep the two tracks separate on the leaderboard.

## Reproduce

LLM track — one model swept over all three difficulties in a single Hydra
multirun (`-m`); one client entry per agent (see `baselines/llm/README.md` for
providers):

```bash
cd baselines/llm
python eval_alem.py -m \
    agent.type=robust_all agent.use_cot=True agent.use_communication=True agent.use_scratchpad=True \
    agent.reasoning=True \
    alem.coordination_difficulty=easy,medium,hard \
    clients.0.client_name=openai clients.1.client_name=openai clients.2.client_name=openai \
    clients.0.model_id=gpt-4o-mini clients.1.model_id=gpt-4o-mini clients.2.model_id=gpt-4o-mini
```

RL track — train **and** evaluate on the same difficulty, one run per difficulty
(repeat for `easy`, `medium`, `hard`):

```bash
cd baselines
python ippo_rnn.py TRAINING_COORDINATION_DIFFICULTY=hard EVAL_DIFFICULTIES=[hard]
```

## Submitting to the leaderboard

To put a result on the [leaderboard](https://alem-world.github.io/leaderboard), follow the step-by-step in [`SUBMISSION.md`](SUBMISSION.md). Report the exact model/algorithm, coordination difficulty, the metrics above, and confirm you used the standard settings in this document.
