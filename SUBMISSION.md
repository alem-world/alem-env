# Submitting to the Alem leaderboard

Alem is an open benchmark — evaluate any LLM agent, MARL policy, or custom harness and submit the result. Entries are listed as **self-reported**; we re-run a representative sample to mark them **✓ verified**.

The canonical settings live in [`EVALUATION.md`](EVALUATION.md); this page is the step-by-step for getting a number onto the [leaderboard](https://alem-world.github.io/leaderboard).

## 1. Install

```bash
git clone https://github.com/alem-world/alem-env
cd alem-env
uv pip install -e ".[baselines-llm]"   # or ".[baselines-rl]" for MARL
```

## 2. Evaluate under the standard protocol

The leaderboard uses **zero-shot, homogeneous 3-agent teams** on the symbolic/text interface (`Alem-Coop-Symbolic`), scored on **Easy / Medium / Hard** *separately*. Episodes use shared seeds (`EVAL_SEED=9999`, episode `i` → seed `9999+i`), so every agent sees the same worlds. Report **≥ 10 seeds** per difficulty (we use 20 for open-weight, 10 for API models) with the mean and a 95% bootstrap CI.

```bash
# LLM track — one model swept over all three difficulties (Hydra multirun)
cd baselines/llm
python eval_alem.py -m \
    agent.type=robust_all agent.use_cot=True agent.use_communication=True agent.use_scratchpad=True \
    agent.reasoning=True \
    alem.coordination_difficulty=easy,medium,hard \
    eval.num_episodes.alem=20 EVAL_SEED=9999 \
    clients.0.client_name=openai clients.1.client_name=openai clients.2.client_name=openai \
    clients.0.model_id=your-model clients.1.model_id=your-model clients.2.model_id=your-model
```

The default `robust_all` harness gives each agent broadcast communication, scratchpad memory, reasoning, and the last 8 turns of history. To benchmark a **different harness**, add an agent under [`baselines/llm/eval_utils/agents/`](baselines/llm/eval_utils/agents/) and say which harness you used — **the model *and* the harness are both part of a submission.**

(MARL track: train **and** evaluate on the same difficulty, one run per difficulty — `python baselines/ippo_rnn.py TRAINING_COORDINATION_DIFFICULTY=hard EVAL_DIFFICULTIES=[hard]`.)

## 3. Collect Base% / Coord.% / Total%

Each run reports normalised episode return as a percentage of the maximum achievable reward, **per category**: **Base%** (66 individual achievements), **Coord.%** (27 coordination achievements) and **Total%** (93). The three are normalised independently — **Total% is not the sum**. The runner logs `Team/ normal_reward_pct_of_max`, `Team/coord_reward_pct_of_max` and `Team/reward_pct_of_max`; aggregate across seeds with a 95% bootstrap CI (we use [`rliable`](https://github.com/google-research/rliable)).

## 4. Format your entry

Add one object matching the leaderboard schema ([`data/leaderboard.json`](https://github.com/alem-world/alem-env)). Each score is `[mean, ci_low, ci_high]`.

```jsonc
{
  "id": "your-model-id",
  "name": "Your Model",
  "config": "harness: robust_all · 20 seeds",
  "type": "open-weight",          // "open-weight" | "proprietary"
  "family": "YourFamily",
  "params": "27B dense",
  "harness_version": "robust_all_v0.1",
  "verified": false,
  "scores": {
    "easy":   {"base": [0,0,0], "coord": [0,0,0], "total": [0,0,0]},
    "medium": {"base": [0,0,0], "coord": [0,0,0], "total": [0,0,0]},
    "hard":   {"base": [0,0,0], "coord": [0,0,0], "total": [0,0,0]}
  }
}
```

## 5. Send it

- **Pull request** (preferred) — add your entry to `data/leaderboard.json` and open a PR. Keeps a public, reviewable record.
- **Issue** — [open an issue](https://github.com/alem-world/alem-env/issues/new) with the JSON if you'd rather not open a PR.
- **Email** — send to <k.tessera@ed.ac.uk> with subject `Alem leaderboard submission`.

Include enough to reproduce: model id / API + date, harness, number of seeds, and any non-default config. Attaching run logs or the `eval_alem.py` debug HTML helps us verify faster. Questions? See [`baselines/llm/README.md`](baselines/llm/README.md) or the [paper](https://arxiv.org/abs/2606.08340).
