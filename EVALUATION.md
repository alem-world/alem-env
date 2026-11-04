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

The leaderboard's **Total% / Coord.% / Base%** are reward-normalised, averaged over
the episodes:

- `eval/<difficulty>/Team/reward_pct_of_max` is **Total%**
- `eval/<difficulty>/Team/coord_reward_pct_of_max` is **Coord.%**
- `eval/<difficulty>/Team/normal_reward_pct_of_max` is **Base%**

Each is a percentage of the maximum achievable *reward* in its category, and each is
normalised independently, so Total% is not the sum or the plain mean of the other
two. It is their blend, weighted by the category reward maxima.

Do not compute these by hand: `scripts/make_submission.py` reads them and produces
the entry, CIs included. The episode json also carries a similarly-named
`Team/*achievement_pct` family, which counts achievements unlocked (out of 66 base /
27 coordination / 93 total) rather than reward earned. It is a useful diagnostic and
worth reporting, but it is **not** the leaderboard column, and the two differ by
roughly a factor of two, so mixing them up silently halves or doubles a score.

Always report alongside the headline numbers:

- `eval/<difficulty>/Team/achievement_pct` and its `coordination_`/`normal_` variants
  (the achievement-count view of the same run)
- `eval/action_parse_rate` (LLM only), the fraction of outputs successfully parsed;
  a low value means the score is throttled by formatting failures, not capability.

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

## Running at scale

A full submission is larger than it looks. 20 episodes x 3 difficulties, episodes
that typically run a few hundred steps, and one LLM call per agent per step, is
on the order of *50,000 model calls*. Typically can take hours, not minutes.

### Resume is automatic, so use it

The evaluator writes one JSON per episode and **skips any episode whose JSON
already exists**. Point `eval.resume_from` at a previous run directory and it
continues from where it stopped:

```bash
python baselines/llm/eval_alem.py \
    eval.resume_from=outputs/alem_eval/<run-dir> \
    ... # same overrides as the original run
```

This is what makes Alem runnable on batch schedulers with a walltime cap: submit
the same job again and it picks up mid-run rather than restarting. A run that is
already complete exits in seconds, so re-submitting is always safe.

Two things to keep true, or you will pool incomparable episodes into one mean:

- resume only into a directory scored with the **same settings**. A changed
  episode count, step cap, seed or harness makes a different experiment;
- resume only with the **same model and weights revision**. Pinning the
  revision (`--revision` on the server, or a specific checkpoint id) is worth
  doing for anything you intend to submit.

A directory name that encodes model, revision and difficulty makes both of these
hard to get wrong by accident.

### `eval.num_workers` is a throughput knob and nothing else

Workers are episodes run concurrently, and each episode has `alem.num_agents`
agents in flight, so:

```
concurrent requests to your endpoint = eval.num_workers x alem.num_agents
```

The default of 4 is deliberately conservative. Raising it is the single largest
speedup available.

What bounds it is your serving setup, not Alem. For a local vLLM server the
ceiling is KV cache: roughly `(GPU memory x gpu-memory-utilization) - weights`,
divided by the per-request cost of `max-model-len` tokens. If the server starts
queueing or runs out of KV, lower `num_workers` first; never shorten
`max-model-len` to buy KV, because a context shorter than the prompt silently
truncates the history the `robust_all` harness depends on, which changes agent
behaviour rather than just speed.

### Record what produced the number

Alongside the score, keep the `alem-env` version
(`python -c "import alem; print(alem.__version__)"`), the exact model id and
weights revision, and the serving engine version. `SUBMISSION.md` asks for the
first of these, and the rest are what make a result reproducible months later.

## Submitting to the leaderboard

To put a result on the [leaderboard](https://alem-world.github.io/leaderboard.html), follow the
step-by-step in [`SUBMISSION.md`](SUBMISSION.md). Report the exact model/algorithm, coordination
difficulty, the metrics above, the `alem` version you evaluated on, and confirm you used the
standard settings in this document.

**Where to open the PR.** The live leaderboard renders from `data/leaderboard.json` in the site
repo, [alem-world/alem-world.github.io](https://github.com/alem-world/alem-world.github.io). Add
your entry to the `homogeneous` list there (or `marl` for the MARL track, `heterogeneous.teams`
for a mixed team) and attach the bundle `make_submission.py` wrote. That is the only copy of the
leaderboard; this repo deliberately does not keep one, so there is nothing to keep in sync.
Prefer not to PR? Email both to <kaleabtessera@gmail.com>.
